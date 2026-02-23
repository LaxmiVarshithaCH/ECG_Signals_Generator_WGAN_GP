# training/gradient_penalty.py

import torch
import torch.autograd as autograd


def compute_gradient_penalty(critic, real_samples, fake_samples, device):

    batch_size = real_samples.size(0)

    alpha = torch.rand(batch_size, 1, 1, device=device)
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates.requires_grad_(True)

    critic_interpolates = critic(interpolates)

    gradients = autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(critic_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)

    penalty = ((gradient_norm - 1) ** 2).mean()

    return penalty