import torch
from torch import nn

    
def q_sample_diffuser(noise_sched, x_start: torch.Tensor, R: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor=None, last=True):
    if noise is None:
        noise = torch.randn_like(x_start)
    timesteps = timesteps.to(x_start.device)
    noisy_images = noise_sched.add_noise(x_start, noise, timesteps)
    if last:
        return noisy_images + R, noise
    else:
        return noisy_images + 0*R, noise


def p_losses_diffuser(noise_sched, model: nn.Module, x_start: torch.Tensor, R: torch.Tensor, timesteps: torch.Tensor, last =True):
    if len(x_start) == 0: 
        return 0
    noise_1 = torch.randn_like(x_start)
    noise_2 = torch.randn_like(x_start)
    x_noisy_1, target_1 = q_sample_diffuser(noise_sched=noise_sched, x_start=x_start, R=R, timesteps=timesteps, noise=noise_1, last =last)
    x_noisy_2, target_2 = q_sample_diffuser(noise_sched=noise_sched, x_start=x_start, R=R, timesteps=timesteps, noise=noise_2, last =last)
    predicted_noise_1 = model(x_noisy_1, timesteps, return_dict=False)[0]
    predicted_noise_2 = model(x_noisy_2, timesteps, return_dict=False)[0]
    loss = 0.5*(target_1-predicted_noise_1-(target_2-predicted_noise_2)).square().sum(dim=(1, 2, 3)).mean(dim=0)
    return loss