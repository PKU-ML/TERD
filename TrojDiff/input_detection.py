import tqdm
import torch
import argparse
from PIL import Image
from torch.distributions import MultivariateNormal
import torchvision.transforms as T


torch.set_printoptions(sci_mode=False)

parser = argparse.ArgumentParser(description=globals()["__doc__"])
parser.add_argument("--path", type=str, required=True)
parser.add_argument("--num_detect", type=int, default=1000)


parser.add_argument("--gamma", type=float, default=0.6)
parser.add_argument("--miu_path", type=str, default="./images/hello_kitty.png")
parser.add_argument("--image_size", type=int, default=32)


args = parser.parse_args()


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 
path=args.path
mu = torch.load(path)["mu"].cuda().detach().view(-1)
gamma = torch.load(path)["gamma"].cuda().detach().view(-1)




def data_transform():
    miu = Image.open(args.miu_path)
    transform = T.Compose([
        T.Resize((args.image_size, args.image_size)),
        T.ToTensor()
    ])
    miu = transform(miu)
    miu = 2 * miu - 1.0
    return miu

miu = data_transform()
miu = miu * (1 - args.gamma)
true_mu = miu
true_gamma = args.gamma


rev_mu = torch.flatten(mu.cuda())
rev_gamma = torch.flatten(gamma.cuda())

benign_mu = torch.flatten(torch.zeros_like(rev_mu).cuda())
benign_gamma = torch.flatten(torch.ones_like(rev_gamma).cuda())


rev_gamma = torch.diag(rev_gamma)
benign_gamma = torch.diag(benign_gamma)
multi_normal_benign = MultivariateNormal(benign_mu, benign_gamma)
multi_normal_poison = MultivariateNormal(rev_mu, rev_gamma)
TPR = 0



# benign input detection
for i in tqdm.tqdm(range(args.num_detect)):
    x = torch.flatten(torch.randn(
                3,args.image_size,args.image_size
            ).cuda())
    prob_1 = multi_normal_benign.log_prob(x)
    prob_2 = multi_normal_poison.log_prob(x)
    if prob_2 < prob_1:
        TPR = TPR + 1





TNR = 0
# backdoor input detection
for i in tqdm.tqdm(range(args.num_detect)):
    x = torch.randn(
            3,args.image_size,args.image_size,
            device=device,
        ).cuda()
    x = true_gamma * x + true_mu.to(device)
    x =torch.flatten(x)
    prob_1 = multi_normal_benign.log_prob(x)
    prob_2 = multi_normal_poison.log_prob(x)
    if prob_2>prob_1:
        TNR = TNR+1


print(TPR/args.num_detect, "percent of benign samples are detected!")
print(TNR/args.num_detect, "percent of backdoor samples are detected!")
       