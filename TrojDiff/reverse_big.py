import argparse
import traceback
import logging
import yaml
import sys
import os
from torch.autograd import Variable
import torch
import numpy as np
import tqdm
from models.diffusion import Model
from datasets import data_transform
import torchvision.transforms as T
from PIL import Image





torch.set_printoptions(sci_mode=False)


def parse_args_and_config():
    def dict2namespace(config):
        namespace = argparse.Namespace()
        for key, value in config.items():
            if isinstance(value, dict):
                new_value = dict2namespace(value)
            else:
                new_value = value
            setattr(namespace, key, new_value)
        return namespace

    parser = argparse.ArgumentParser(description=globals()["__doc__"])


    parser.add_argument(
        "--dataset", type=str, required=True
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "--exp", type=str, default="exp", help="Path for saving running related data."
    )
    parser.add_argument(
        "--doc",
        type=str,
        default='ddpm',
        help="A string for documentation purpose. "
        "Will be the name of the log folder.",
    )
    parser.add_argument("--grad_clip",type=float,default=1.0)
    parser.add_argument(
        "--comment", type=str, default="", help="A string for experiment comment"
    )
    parser.add_argument(
        "--verbose",
        type=str,
        default="info",
        help="Verbose level: info | debug | warning | critical",
    )
    parser.add_argument("--test", action="store_true", help="Whether to test the model")
    parser.add_argument("--fid", action="store_true")
    parser.add_argument("--interpolation", action="store_true")
    parser.add_argument(
        "--resume_training", action="store_true", help="Whether to resume training"
    )
    parser.add_argument(
        "-i",
        "--image_folder",
        type=str,
        default="images",
        help="The folder name of samples",
    )
    parser.add_argument(
        "--ni",
        action="store_true",
        help="No interaction. Suitable for Slurm Job launcher",
    )
    parser.add_argument("--use_pretrained", action="store_true")



    parser.add_argument(
        "--skip_type",
        type=str,
        default="quad",
        help="skip according to (uniform or quadratic)",
    )

    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="eta used to control the variances of sigma",
    )
    parser.add_argument("--sequence", action="store_true")


    ###############################################################
    #            Hyperparameter for reverse engineering           #
    ###############################################################

    parser.add_argument('--weight_decay', type=float, default=5e-5, help="lambda")
    parser.add_argument('--lr', type=float, default=0.5, help="Learning rate for optimization mu")
    parser.add_argument('--lr2', type=float, default=0.002, help="Learning rate for optimization gamma")
    parser.add_argument("--timesteps", type=int, default=10, help="DDIM steps")
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument("--out-dir", type=str,default="./log", help="Path to save the reversed trigger.")
    parser.add_argument("--iteration", type=int, default=3000, help="Random seed")
    parser.add_argument("--batch-size", type=int, default=16, help="batch_size of reverse engineer")

    ###############################################################
    #            Hyperparameter for reverse engineering
    ################################################################

    # attack
    parser.add_argument('--cond_prob', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=None)
    parser.add_argument('--target_label', type=int, default=7)
    parser.add_argument('--miu_path', type=str, default='./images/hello_kitty.png')
    parser.add_argument('--total_n_samples', type=int, default=50000)
    parser.add_argument('--trigger_type', type=str, default='blend')
    parser.add_argument('--patch_size', type=int, default=3)
    args = parser.parse_args()
    # parse config file
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def generalized_steps_bd(x, seq, model, b, miu, gamma, args, **kwargs):
    def compute_alpha(beta, t):
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]
    for i, j in zip(reversed(seq), reversed(seq_next)):
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        at = compute_alpha(b, t.long())
        at_next = compute_alpha(b, next_t.long())
        xt = xs[-1].to('cuda')
        et = model(xt, t)

        batch, device = xt.shape[0], xt.device
        miu_ = torch.stack([miu.to(device)] * batch)

        x0_t = (xt - et * (1 - at).sqrt() * gamma - miu_ * (1 - at).sqrt()) / at.sqrt()

        x0_preds.append(x0_t)

        c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
        )
        c2 = ((1 - at_next) - c1 ** 2).sqrt()
        xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(
            x) * gamma + c2 * et * gamma + miu_ * (1 - at_next).sqrt()
        xs.append(xt_next)
    return xs, x0_preds


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)
    if beta_schedule == "quad":
        betas = (
                np.linspace(
                    beta_start ** 0.5,
                    beta_end ** 0.5,
                    num_diffusion_timesteps,
                    dtype=np.float64,
                )
                ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def backdoor_reconstruct_loss(model,
                          x0: torch.Tensor,
                          gamma: torch.Tensor,
                          t: torch.LongTensor,
                          e1: torch.Tensor,
                          e2: torch.Tensor,
                          b: torch.Tensor,
                          miu: torch.Tensor,
                          keepdim=False,
                          surrogate=True):
    batch, device = x0.shape[0], x0.device
    miu_ = torch.stack([miu.to(device)] * batch)  # (batch,3,32,32)
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    if surrogate:
        x_ = x0 * a.sqrt() + e1 * (1.0 - a).sqrt() * gamma + miu_
        x_1 = x0 * a.sqrt() + e2 * (1.0 - a).sqrt() * gamma + miu_
    else:
        x_ = x0 * a.sqrt() + e1 * (1.0 - a).sqrt() * gamma
        x_1 = x0 * a.sqrt() + e2 * (1.0 - a).sqrt() * gamma
    x_add = x_
    x_add_1 = x_1
    t_add = t
    e_add = e1
    e_add_1 = e2
    x = x_add
    x_1 = x_add_1
    t = t_add
    e = e_add
    e_1 = e_add_1
    output = model(x, t.float())
    output_1 = model(x_1, t.float())
    if keepdim:
        return 0.5*(e - output-(e_1-output_1)).square().sum(dim=(1, 2, 3))
    else:
        return 0.5*(e - output-(e_1-output_1)).square().sum(dim=(1, 2, 3)).mean(dim=0)






class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
                betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

        # attack
        miu = Image.open(args.miu_path)

        transform = T.Compose([
            T.Resize((config.data.image_size, config.data.image_size)),
            T.ToTensor()
        ])
        # Vanilla mu
        miu = transform(miu)  # [0,1]
        miu = data_transform(self.config, miu)  # [-1,1]
        miu = miu * (1 - args.gamma)  # [-0.5,0.5]
        self.miu = miu  # (3,32,32)

        k_t = torch.randn_like(betas)
        for ii in range(config.diffusion.num_diffusion_timesteps):
            tmp_sum = torch.sqrt(1. - alphas_cumprod[ii])
            tmp_alphas = torch.flip(alphas[:ii + 1], [0])
            for jj in range(1, ii + 1):
                tmp_sum -= k_t[ii - jj] * torch.sqrt(torch.prod(tmp_alphas[:jj]))
            k_t[ii] = tmp_sum
        coef_miu = torch.sqrt(1. - alphas_cumprod_prev) * betas - (1. - alphas_cumprod_prev) * torch.sqrt(alphas) * k_t
        self.coef_miu = coef_miu



    def reverse(self, args):
        model = Model(self.config)
        print("Load "+ self.args.trigger_type + "checkpoint.")
        states = torch.load(self.args.checkpoint,map_location=self.config.device)
        model = model.to(self.device)
        print("Loading model...")
        model = torch.nn.DataParallel(model)

        model.load_state_dict(states[0], strict=True)



        model.eval()
        config = self.config
        iterations = self.args.iteration
        batch_size = self.args.batch_size
        mu = Variable(
            -torch.rand(config.data.channels, config.data.image_size, config.data.image_size, device=self.device),
            requires_grad=True)

        gamma = Variable(
            torch.zeros(config.data.channels, config.data.image_size, config.data.image_size, device=self.device),
            requires_grad=True)


        optim = torch.optim.SGD([mu], lr=self.args.lr,  weight_decay=0)
        optim_1 = torch.optim.SGD([gamma], lr=self.args.lr2, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, iterations)
        scheduler_1 = torch.optim.lr_scheduler.CosineAnnealingLR(optim_1, iterations)

        self.args.out_dir = self.args.out_dir+"log_"+str(self.args.weight_decay)+"_"+str(self.args.timesteps)\
                            +"_"+str(self.args.batch_size)+"_"+str(self.args.lr)+"_"+str(self.args.lr2)\
                            +"_"+str(self.args.iteration)+"/"

        os.makedirs(self.args.out_dir, exist_ok=True)




        for _ in tqdm.tqdm(
                range(iterations), desc="Trigger Estimation."
        ):
            n = batch_size
            x = torch.randn(
                n,
                config.data.channels,
                config.data.image_size,
                config.data.image_size,
                device=self.device,
            )
            batch_miu = torch.stack([mu.to(self.device)] * n)  # (batch,3,32,32)
            batch_gamma = torch.stack([gamma.to(self.device)] * n)  # (batch,3,32,32)
            x = batch_gamma * x + batch_miu

            #################################################
            #     Reversed loss for trigger estimation      #
            #################################################

            x = torch.randn_like(x, device=self.device)
            e1 = torch.randn_like(x, device=self.device)
            e2 = torch.randn_like(x, device=self.device)
            b = self.betas
            t = torch.randint(
                low=self.num_timesteps-10, high=self.num_timesteps, size=(n,), device=self.device
            ).to(self.device)
            loss_update = backdoor_reconstruct_loss(model, x, gamma, t, e1, e2, b, mu, surrogate=True) -self.args.weight_decay*torch.norm(mu, p=1)

            optim.zero_grad()
            optim_1.zero_grad()
            loss_update.backward()

            optim.step()
            optim_1.step()
            gamma.data.clip_(min=0)

            scheduler.step()
            scheduler_1.step()
            torch.save({"mu": mu, "gamma": gamma}, os.path.join(self.args.out_dir, "reverse.pkl"))



        optim = torch.optim.SGD([mu], lr=self.args.lr,weight_decay=0, momentum=0.9)
        optim_1 = torch.optim.SGD([gamma], lr=self.args.lr2,weight_decay=0, momentum=0.9)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, iterations//3)
        scheduler_1 = torch.optim.lr_scheduler.CosineAnnealingLR(optim_1, iterations//3)


        for _ in tqdm.tqdm(
                range(iterations, int(iterations*4/3)), desc="Trigger Refinement."
        ):
            n = batch_size
            x = torch.randn(
                n,
                config.data.channels,
                config.data.image_size,
                config.data.image_size,
                device=self.device,
            )
            batch_miu = torch.stack([mu.to(self.device)] * n)  # (batch,3,32,32)
            batch_gamma = torch.stack([gamma.to(self.device)] * n)  # (batch,3,32,32)
            x = batch_gamma * x + batch_miu

            #################################
            #      Generate  image          #
            #################################
            seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
            )
            seq = [int(s) for s in list(seq)]
            xs = generalized_steps_bd(x, seq, model, self.betas, mu, gamma, self.args,
                                      eta=self.args.eta)
            x = xs
            x = x[0][-1]

            #################################################
            #       Reversed loss for trigger refinement    #
            #################################################

            e1 = torch.randn_like(x, device=self.device)
            e2 = torch.randn_like(x, device=self.device)
            b = self.betas
            t = torch.randint(low=self.num_timesteps-10, high=self.num_timesteps, size=(n,), device=self.device)

            loss_1 = backdoor_reconstruct_loss(model, x, gamma, t, e1, e2,  b, mu, surrogate=True)
            e1 = torch.randn_like(x, device=self.device)
            e2 = torch.randn_like(x, device=self.device)
            b = self.betas
            t = torch.randint(low=0, high=10, size=(n,), device=self.device)
            loss_2 = backdoor_reconstruct_loss(model, x, gamma, t, e1, e2,  b, mu, surrogate=False)
            loss_update = (loss_1+loss_2)/2 -self.args.weight_decay*torch.norm(mu, p=1)

            optim.zero_grad()
            optim_1.zero_grad()
            loss_update.backward()

            torch.nn.utils.clip_grad_norm_(
                [mu, gamma], self.config.optim.grad_clip)
            optim.step()
            optim_1.step()

            gamma.data.clip_(min=0)
            scheduler.step()
            scheduler_1.step()
            torch.save({"mu": mu, "gamma": gamma}, os.path.join(self.args.out_dir, "reverse.pkl"))



def main():
    args, config = parse_args_and_config()
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))
    try:
        runner = Diffusion(args, config)
        runner.reverse(args)
    except Exception:
        logging.error(traceback.format_exc())
    return 0


if __name__ == "__main__":
    sys.exit(main())
