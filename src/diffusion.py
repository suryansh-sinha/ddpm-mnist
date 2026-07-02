import torch

class LinearNoiseScheduler:
    def __init__(self, num_timesteps, beta_start, beta_end):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        # All these tensors have dimension --> [1000] if num_timesteps = 1000
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1. - self.alpha_bars)

    def add_noise(self, x_0, noise, t):
        # sqrt_alpha_bars[t] has shape [batch_size] and we need to multiply it with [batch_size, 1, 28, 28] image.
        # for broadcasting to work correctly, we are adding dimensions to make it [batch_size, 1, 1, 1]
        sqrt_alpha_bars = self.sqrt_alpha_bars[t].reshape([-1, 1, 1, 1]).to(x_0.device)   # Bx1x1x1
        sqrt_one_minus_alpha_bars = self.sqrt_one_minus_alpha_bars[t].reshape([-1, 1, 1, 1]).to(x_0.device)   # Bx1x1x1

        # the direct formula to get x_t from x_0
        return sqrt_alpha_bars*x_0 + sqrt_one_minus_alpha_bars*noise
    
    def sampling(self):
        pass