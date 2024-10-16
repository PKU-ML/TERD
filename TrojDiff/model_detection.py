import torch
import argparse
torch.set_printoptions(sci_mode=False)
parser = argparse.ArgumentParser(description=globals()["__doc__"])
parser.add_argument("--path", type=str, required=True)
args = parser.parse_args()
path=args.path


mu = torch.load(path)["mu"].cuda().detach().view(-1)
mu = torch.flatten(mu.cuda().detach())
gamma = torch.load(path)["gamma"].cuda().detach().view(-1)
gamma = torch.flatten(gamma.cuda().detach())
kl_divergence = (-torch.log(gamma) + (gamma * gamma + mu * mu-1) / 2)

# N_m and N_v are simply coefficient for normalization. You can also set it to zero to perform model detection.
N_m = -0.4
N_v = 0.003
M_r = kl_divergence.mean(dim=0)-N_m
V_r = (kl_divergence - kl_divergence.mean(dim=0)).square().mean(dim=0)-N_v
print("M_r:", M_r)
print("V_r:", V_r)