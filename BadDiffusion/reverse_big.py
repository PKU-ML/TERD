from dataclasses import dataclass
import argparse
import os
import json
from typing import Dict, Union
from torch.autograd import Variable
import torch
import numpy as np
import tqdm as tqdm1
from dataset import DatasetLoader, Backdoor

MODE_TRAIN: str = 'train'
MODE_RESUME: str = 'resume'
MODE_SAMPLING: str = 'sampling'
MODE_MEASURE: str = 'measure'
MODE_TRAIN_MEASURE: str = 'train+measure'
DEFAULT_PROJECT: str = "Default"
DEFAULT_BATCH: int = 512
DEFAULT_EVAL_MAX_BATCH: int = 1024
DEFAULT_EPOCH: int = 50
DEFAULT_LEARNING_RATE: float = None
DEFAULT_LEARNING_RATE_32: float = 2e-4
DEFAULT_LEARNING_RATE_256: float = 8e-5
DEFAULT_CLEAN_RATE: float = 1.0
DEFAULT_POISON_RATE: float = 0.007
DEFAULT_TRIGGER: str = Backdoor.TRIGGER_BOX_14
DEFAULT_TARGET: str = Backdoor.TARGET_CORNER
DEFAULT_DATASET_LOAD_MODE: str = DatasetLoader.MODE_FIXED
DEFAULT_GPU = '0'
DEFAULT_CKPT: str = None
DEFAULT_OVERWRITE: bool = False
DEFAULT_POSTFIX: str = ""
DEFAULT_FCLIP: str = 'o'
DEFAULT_SAVE_IMAGE_EPOCHS: int = 10
DEFAULT_SAVE_MODEL_EPOCHS: int = 5
DEFAULT_IS_SAVE_ALL_MODEL_EPOCHS: bool = False
DEFAULT_SAMPLE_EPOCH: int = None
DEFAULT_RESULT: int = '.'

NOT_MODE_TRAIN_OPTS = ['sample_ep']
NOT_MODE_TRAIN_MEASURE_OPTS = ['sample_ep']
MODE_RESUME_OPTS = ['project', 'mode', 'gpu', 'ckpt']
MODE_SAMPLING_OPTS = ['project', 'mode', 'eval_max_batch', 'gpu', 'fclip', 'ckpt', 'sample_ep']
MODE_MEASURE_OPTS = ['project', 'mode', 'eval_max_batch', 'gpu', 'fclip', 'ckpt', 'sample_ep', 'ddim',
                     'num_inference_steps']
# IGNORE_ARGS = ['overwrite']
IGNORE_ARGS = ['overwrite', 'is_save_all_model_epochs']


def parse_args():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--project', '-pj', required=False, type=str, default= "Reverse", help='Project name')
    parser.add_argument('--mode', '-m', required=False, type=str, help='Train or test the model', default=MODE_MEASURE,
                        choices=[MODE_TRAIN, MODE_MEASURE, MODE_RESUME, MODE_SAMPLING, MODE_MEASURE, MODE_TRAIN_MEASURE])
    parser.add_argument('--dataset', '-ds', type=str, help='Training dataset',
                        choices=[DatasetLoader.MNIST, DatasetLoader.CIFAR10, DatasetLoader.CELEBA,
                                 DatasetLoader.CELEBA_HQ])
    parser.add_argument('--batch', '-b', type=int, help=f"Batch size, default for train: {DEFAULT_BATCH}")
    parser.add_argument('--eval_max_batch', '-eb', type=int,
                        help=f"Batch size of sampling, default for train: {DEFAULT_EVAL_MAX_BATCH}")
    parser.add_argument('--epoch', '-e', type=int, help=f"Epoch num, default for train: {DEFAULT_EPOCH}")
    parser.add_argument('--learning_rate', '-lr', type=float,
                        help=f"Learning rate, default for 32 * 32 image: {DEFAULT_LEARNING_RATE_32}, default for larger images: {DEFAULT_LEARNING_RATE_256}")
    parser.add_argument('--clean_rate', '-cr', type=float, help=f"Clean rate, default for train: {DEFAULT_CLEAN_RATE}")
    parser.add_argument('--poison_rate', '-pr', type=float,
                        help=f"Poison rate, default for train: {DEFAULT_POISON_RATE}")
    parser.add_argument('--clip_norm', default= 0.2, type=int, help="Norm for clipping.")
    parser.add_argument('--trigger', '-tr', type=str, help=f"Trigger pattern, default for train: {DEFAULT_TRIGGER}")
    parser.add_argument('--target', '-ta', type=str, help=f"Target pattern, default for train: {DEFAULT_TARGET}")
    parser.add_argument('--dataset_load_mode', '-dlm', type=str,
                        help=f"Mode of loading dataset, default for train: {DEFAULT_DATASET_LOAD_MODE}",
                        choices=[DatasetLoader.MODE_FIXED, DatasetLoader.MODE_FLEX])
    parser.add_argument('--gpu', '-g', type=str, help=f"GPU usage, default for train/resume: {DEFAULT_GPU}")
    parser.add_argument('--ckpt', '-c', type=str, help=f"Load from the checkpoint, default: {DEFAULT_CKPT}")
    parser.add_argument('--overwrite', '-o', action='store_true',
                        help=f"Overwrite the existed training result or not, default for train/resume: {DEFAULT_CKPT}")
    parser.add_argument('--postfix', '-p', type=str,
                        help=f"Postfix of the name of the result folder, default for train/resume: {DEFAULT_POSTFIX}")
    parser.add_argument('--fclip', '-fc', type=str,
                        help=f"Force to clip in each step or not during sampling/measure, default for train/resume: {DEFAULT_FCLIP}",
                        choices=['w', 'o'])
    parser.add_argument('--save_image_epochs', '-sie', type=int,
                        help=f"Save sampled image per epochs, default: {DEFAULT_SAVE_IMAGE_EPOCHS}")
    parser.add_argument('--save_model_epochs', '-sme', type=int,
                        help=f"Save model per epochs, default: {DEFAULT_SAVE_MODEL_EPOCHS}")
    parser.add_argument('--is_save_all_model_epochs', '-isame', action='store_true', help=f"")
    parser.add_argument('--sample_ep', '-se', type=int,
                        help=f"Select i-th epoch to sample/measure, if no specify, use the lastest saved model, default: {DEFAULT_SAMPLE_EPOCH}")
    parser.add_argument('--result', '-res', type=str, help=f"Output file path, default: {DEFAULT_RESULT}")
    parser.add_argument('--num_inference_steps', default=50, type=int, help="Number of time steps.")
    parser.add_argument('--ddim', action='store_true', help="Whether to adopt the ddim sampling")

    # hyperparmeter for reverse engineering
    parser.add_argument('--weight_decay', type=float, default=5e-4, help="Trade-off coefficient.")
    parser.add_argument('--lr', type=float, default=0.5, help="Learning rate for reversed engineering")
    parser.add_argument("--iteration", type=int, default=3000, help="Iterations for Trigger Estimation")
    parser.add_argument("--out-dir", type=str, default="./log/log",
                        help="Path to the config file")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch_size of reverse engineer")
    parser.add_argument('--num_steps', default=10, type=int, help="Number of steps for DDIM sampler.")
    # hyperparmeter for reverse engineering


    args = parser.parse_args()
    return args


@dataclass
class TrainingConfig:
    project: str = DEFAULT_PROJECT
    batch: int = DEFAULT_BATCH
    epoch: int = DEFAULT_EPOCH
    eval_max_batch: int = DEFAULT_EVAL_MAX_BATCH
    learning_rate: float = DEFAULT_LEARNING_RATE
    clean_rate: float = DEFAULT_CLEAN_RATE
    poison_rate: float = DEFAULT_POISON_RATE
    trigger: str = DEFAULT_TRIGGER
    target: str = DEFAULT_TARGET
    dataset_load_mode: str = DEFAULT_DATASET_LOAD_MODE
    gpu: str = DEFAULT_GPU
    ckpt: str = DEFAULT_CKPT
    overwrite: bool = DEFAULT_OVERWRITE
    postfix: str = DEFAULT_POSTFIX
    fclip: str = DEFAULT_FCLIP
    save_image_epochs: int = DEFAULT_SAVE_IMAGE_EPOCHS
    save_model_epochs: int = DEFAULT_SAVE_MODEL_EPOCHS
    is_save_all_model_epochs: bool = DEFAULT_IS_SAVE_ALL_MODEL_EPOCHS
    sample_ep: int = DEFAULT_SAMPLE_EPOCH
    result: str = DEFAULT_RESULT

    eval_sample_n: int = 16  # how many images to sample during evaluation
    measure_sample_n: int = 16
    batch_32: int = 128
    batch_256: int = 64
    gradient_accumulation_steps: int = 1
    learning_rate_32_scratch: float = 2e-4
    learning_rate_256_scratch: float = 2e-5
    lr_warmup_steps: int = 500
    # save_image_epochs: int = 1
    mixed_precision: str = 'fp16'  # `no` for float32, `fp16` for automatic mixed precision
    # mixed_precision: str = 'no'  # `no` for float32, `fp16` for automatic mixed precision

    push_to_hub: bool = False  # whether to upload the saved model to the HF Hub
    hub_private_repo: bool = False
    overwrite_output_dir: bool = True  # overwrite the old model when re-running the notebook
    seed: int = 0
    dataset_path: str = 'datasets'
    ckpt_dir: str = 'ckpt'
    data_ckpt_dir: str = 'data.ckpt'
    ep_model_dir: str = 'epochs'
    ckpt_path: str = None
    data_ckpt_path: str = None


def naming_fn(config: TrainingConfig):
    add_on: str = ""
    # add_on += "_clip" if config.clip else ""
    add_on += f"_{config.postfix}" if config.postfix else ""
    return f'res_{config.ckpt}_{config.dataset}_ep{config.epoch}_c{config.clean_rate}_p{config.poison_rate}_{config.trigger}-{config.target}{add_on}'


def read_json(args: argparse.Namespace, file: str):
    with open(os.path.join(args.ckpt, file), "r") as f:
        return json.load(f)


def write_json(content: Dict, config: argparse.Namespace, file: str):
    with open(os.path.join(config.output_dir, file), "w") as f:
        return json.dump(content, f, indent=2)


def setup():
    args_file: str = "args.json"
    config_file: str = "config.json"
    sampling_file: str = "sampling.json"
    measure_file: str = "measure.json"

    args: argparse.Namespace = parse_args()
    config: TrainingConfig = TrainingConfig()
    args_data: Dict = {}

    if args.mode == MODE_RESUME or args.mode == MODE_SAMPLING or args.mode == MODE_MEASURE:
        with open(os.path.join(args.ckpt, args_file), "r") as f:
            args_data = json.load(f)

        for key, value in args_data.items():
            if value != None:
                setattr(config, key, value)
        setattr(config, "output_dir", args.ckpt)

    for key, value in args.__dict__.items():
        if args.mode == MODE_TRAIN and (key not in NOT_MODE_TRAIN_OPTS) and value != None:
            setattr(config, key, value)
        elif args.mode == MODE_TRAIN_MEASURE and (key not in NOT_MODE_TRAIN_MEASURE_OPTS) and value != None:
            setattr(config, key, value)
        elif args.mode == MODE_RESUME and key in MODE_RESUME_OPTS and value != None:
            setattr(config, key, value)
        elif args.mode == MODE_SAMPLING and key in MODE_SAMPLING_OPTS and value != None:
            setattr(config, key, value)
        elif args.mode == MODE_MEASURE and key in MODE_MEASURE_OPTS and value != None:
            setattr(config, key, value)

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", config.gpu)

    print(f"PyTorch detected number of availabel devices: {torch.cuda.device_count()}")
    setattr(config, "device_ids", [int(i) for i in range(len(config.gpu.split(',')))])

    # sample_ep options
    if isinstance(config.sample_ep, int):
        if config.sample_ep < 0:
            config.sample_ep = None

    # Clip option
    if config.fclip == 'w':
        setattr(config, "clip", True)
    elif config.fclip == 'o':
        setattr(config, "clip", False)
    else:
        setattr(config, "clip", None)

    # Determine gradient accumulation & Learning Rate
    bs = 0
    if config.dataset in [DatasetLoader.CIFAR10, DatasetLoader.MNIST]:
        bs = config.batch_32
        if config.learning_rate == None:
            if config.ckpt == None:
                config.learning_rate = config.learning_rate_32_scratch
            else:
                config.learning_rate = DEFAULT_LEARNING_RATE_32
    elif config.dataset in [DatasetLoader.CELEBA, DatasetLoader.CELEBA_HQ, DatasetLoader.LSUN_CHURCH,
                            DatasetLoader.LSUN_BEDROOM]:
        bs = config.batch_256
        if config.learning_rate == None:
            if config.ckpt == None:
                config.learning_rate = config.learning_rate_256_scratch
            else:
                config.learning_rate = DEFAULT_LEARNING_RATE_256
    else:
        raise NotImplementedError()
    if bs % config.batch != 0:
        raise ValueError(f"batch size {config.batch} should be divisible to {bs} for dataset {config.dataset}")
    if bs < config.batch:
        raise ValueError(f"batch size {config.batch} should be smaller or equal to {bs} for dataset {config.dataset}")
    config.gradient_accumulation_steps = int(bs // config.batch)

    if args.mode == MODE_TRAIN or args.mode == MODE_TRAIN_MEASURE:
        setattr(config, "output_dir", os.path.join(config.result, naming_fn(config=config)))

    print(f"MODE: {config.mode}")
    if config.mode == MODE_TRAIN or args.mode == MODE_TRAIN_MEASURE:
        if not config.overwrite and os.path.isdir(config.output_dir):
            raise ValueError(
                f"Output directory: {config.output_dir} has already been created, please set overwrite flag --overwrite or -o")

        os.makedirs(config.output_dir, exist_ok=True)

        write_json(content=vars(args), config=config, file=args_file)
        write_json(content=config.__dict__, config=config, file=config_file)
    elif config.mode == MODE_SAMPLING:
        write_json(content=config.__dict__, config=config, file=sampling_file)
    elif config.mode == MODE_MEASURE or args.mode == MODE_TRAIN_MEASURE:
        write_json(content=config.__dict__, config=config, file=measure_file)
    elif config.mode == MODE_RESUME:
        pass
    else:
        raise NotImplementedError(f"Mode: {config.mode} isn't defined")

    if config.ckpt_path == None:
        config.ckpt_path = os.path.join(config.output_dir, config.ckpt_dir)
        config.data_ckpt_path = os.path.join(config.output_dir, config.data_ckpt_dir)
        os.makedirs(config.ckpt_path, exist_ok=True)

    print(f"Argument Final: {config.__dict__}")
    return config, args


config, args = setup()
"""## Config

For convenience, we define a configuration grouping all the training hyperparameters. This would be similar to the arguments used for a [training script](https://github.com/huggingface/diffusers/tree/main/examples).
Here we choose reasonable defaults for hyperparameters like `num_epochs`, `learning_rate`, `lr_warmup_steps`, but feel free to adjust them if you train on your own dataset. For example, `num_epochs` can be increased to 100 for better visual quality.
"""

from torch import nn
from accelerate import Accelerator
from diffusers.hub_utils import init_git_repo

from diffusers import DDIMScheduler
from reverse_pipeline import DDIMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from model import DiffuserModelSched
from reverse_loss import p_losses_diffuser


def get_accelerator(config: TrainingConfig):
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with=["tensorboard"],
        logging_dir=os.path.join(config.output_dir, "logs")
    )
    return accelerator


def init_tracker(config: TrainingConfig, accelerator: Accelerator):
    tracked_config = {}
    for key, val in config.__dict__.items():
        if isinstance(val, int) or isinstance(val, float) or isinstance(val, str) or isinstance(val,
                                                                                                bool) or isinstance(val,
                                                                                                                    torch.Tensor):
            tracked_config[key] = val
    accelerator.init_trackers(config.project, config=tracked_config)


def get_data_loader(config: TrainingConfig):
    ds_root = os.path.join(config.dataset_path)
    dsl = DatasetLoader(root=ds_root, name=config.dataset, batch_size=config.batch).set_poison(
        trigger_type=config.trigger, target_type=config.target, clean_rate=config.clean_rate,
        poison_rate=config.poison_rate).prepare_dataset(mode=config.dataset_load_mode)
    print(f"datasetloader len: {len(dsl)}")
    return dsl


def get_repo(config: TrainingConfig, accelerator: Accelerator):
    repo = None
    if accelerator.is_main_process:
        if config.push_to_hub:
            repo = init_git_repo(config, at_init=True)
        init_tracker(config=config, accelerator=accelerator)
    return repo


def get_model_optim_sched(config: TrainingConfig, accelerator: Accelerator, dataset_loader: DatasetLoader):
    print(config.ckpt)
    if config.ckpt != None:
        if config.sample_ep != None and config.mode in [MODE_MEASURE, MODE_SAMPLING]:
            ep_model_path = get_ep_model_path(config=config, dir=config.ckpt, epoch=config.sample_ep)
            model, noise_sched = DiffuserModelSched.get_pretrained(ckpt=ep_model_path, clip_sample=config.clip)
        else:
            print(config.ckpt)
            model, noise_sched = DiffuserModelSched.get_pretrained(ckpt=config.ckpt, clip_sample=config.clip)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    else:
        model, noise_sched = DiffuserModelSched.get_model_sched(image_size=dataset_loader.image_size,
                                                                channels=dataset_loader.channel,
                                                                model_type=DiffuserModelSched.MODEL_DEFAULT,
                                                                clip_sample=config.clip)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    model = nn.DataParallel(model, device_ids=config.device_ids)
    lr_sched = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(dataset_loader.num_batch * config.epoch),
    )
    cur_epoch = cur_step = 0
    accelerator.register_for_checkpointing(model, optimizer, lr_sched)
    if config.mode == MODE_RESUME:
        if config.ckpt == None:
            raise ValueError(f"Argument 'ckpt' shouldn't be None for resume mode")
        accelerator.load_state(config.ckpt_path)
        data_ckpt = torch.load(config.data_ckpt_path)
        cur_epoch = data_ckpt['epoch']
        cur_step = data_ckpt['step']
    return model, optimizer, lr_sched, noise_sched, cur_epoch, cur_step


def init_train(config: TrainingConfig, dataset_loader: DatasetLoader):
    # Initialize accelerator and tensorboard logging
    accelerator = get_accelerator(config=config)
    repo = get_repo(config=config, accelerator=accelerator)
    model, optimizer, lr_sched, noise_sched, cur_epoch, cur_step = get_model_optim_sched(config=config,
                                                                                         accelerator=accelerator,
                                                                                         dataset_loader=dataset_loader)
    dataloader = dataset_loader.get_dataloader()
    model, optimizer, dataloader, lr_sched = accelerator.prepare(
        model, optimizer, dataloader, lr_sched
    )
    return accelerator, repo, model, noise_sched, optimizer, dataloader, lr_sched, cur_epoch, cur_step



def reverse(model, dataset_loader, ddpm_noise_sched, config: TrainingConfig, folder_name: Union[int, str], pipeline,
            args=None):
    folder_path_ls = [config.output_dir, folder_name]
    if config.sample_ep != None:
        folder_path_ls += [f"ep{config.sample_ep}"]
    mu = Variable(
        -torch.rand(pipeline.unet.in_channels, pipeline.unet.sample_size, pipeline.unet.sample_size).cuda(),
        requires_grad=True)
    optim = torch.optim.SGD([mu], lr=args.lr, weight_decay=0)
    iterations = args.iteration
    batch_size = args.batch_size
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, iterations)
    args.out_dir = args.out_dir + args.project + "/log_" + str(args.weight_decay) + "_" + str(
        args.num_steps) + "_" + str(
        args.iteration) + "_" + str(args.batch_size) + "_" + str(args.lr)  + "/"
    os.makedirs(args.out_dir, exist_ok=True)
    model.eval()

    for _ in tqdm1.tqdm(
            range(args.iteration), desc="Trigger Estimation"
    ):
        #################################################
        #       Reversed loss for Trigger Estimation    #
        #################################################
        bs = batch_size
        timesteps = torch.randint(noise_sched.num_train_timesteps - 10, noise_sched.num_train_timesteps,
                                  (bs,)).long().cuda()
        fake_image = torch.randn(
            (batch_size, pipeline.unet.in_channels, pipeline.unet.sample_size, pipeline.unet.sample_size)).cuda()

        loss = p_losses_diffuser(ddpm_noise_sched, model=model, x_start=fake_image, R=mu, timesteps=timesteps)
        loss_update = loss - args.weight_decay * torch.norm(mu, p=1)
        optim.zero_grad()
        loss_update.backward()
        optim.step()
        scheduler.step()
        torch.save({"mu": mu}, os.path.join(args.out_dir, "reverse.pkl"))


    optim = torch.optim.SGD([mu], lr=args.lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, iterations // 3)

    for _ in tqdm1.tqdm(
            range(iterations, int(iterations * 4 / 3)), desc="Trigger Refinement"
    ):
        n = batch_size
        noise = torch.randn(
            (batch_size, pipeline.unet.in_channels, pipeline.unet.sample_size, pipeline.unet.sample_size),
        ).cuda()
        batch_miu = torch.stack([mu.cuda()] * n)  # (batch,3,32,32)
        x = noise + batch_miu
        #################################
        #      Generate  image          #
        #################################
        generate_image = pipeline(
            batch_size=args.batch_size,
            generator=None,
            init=x,
            output_type=None,
            num_inference_steps=args.num_steps,
        )

        #################################################
        #       Reversed loss for trigger refinement    #
        #################################################

        bs = generate_image.shape[0]
        timesteps = torch.randint(noise_sched.num_train_timesteps - 10, noise_sched.num_train_timesteps,
                                  (bs,)).long().cuda()
        loss_1 = p_losses_diffuser(ddpm_noise_sched, model=model, x_start=generate_image, R=mu, timesteps=timesteps, last=True)
        timesteps = torch.randint(0, 10, (bs,)).long().cuda()
        loss_2 = p_losses_diffuser(ddpm_noise_sched, model=model, x_start=generate_image, R=mu, timesteps=timesteps, last=False)
        loss_update = (loss_1+loss_2)/2 - args.weight_decay * torch.norm(mu, p=1)
        optim.zero_grad()
        loss_update.backward()
        torch.nn.utils.clip_grad_norm_(
            [mu], args.clip_norm)
        optim.step()
        scheduler.step()
        torch.save({"mu": mu}, os.path.join(args.out_dir, "reverse.pkl"))




def get_ep_model_path(config: TrainingConfig, dir: Union[str, os.PathLike], epoch: int):
    return os.path.join(dir, config.ep_model_dir, f"ep{epoch}")

"""
Let's reverse the trigger!
"""
dsl = get_data_loader(config=config)
accelerator, repo, model, noise_sched, optimizer, dataloader, lr_sched, cur_epoch, cur_step = init_train(config=config, dataset_loader=dsl)
ddim_noise_sched = DDIMScheduler(beta_end=0.02, beta_schedule="linear", beta_start=0.0001,
                                 num_train_timesteps=1000)
ddpm_noise_sched = noise_sched
pipeline = DDIMPipeline(unet=accelerator.unwrap_model(model), scheduler=ddim_noise_sched)
reverse(config=config, dataset_loader=dsl, ddpm_noise_sched=ddpm_noise_sched, model=model, folder_name='measure',
            pipeline=pipeline, args=args)
accelerator.end_training()