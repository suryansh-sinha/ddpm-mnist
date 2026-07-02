import torch

class LinearNoiseScheduler:
    def __init__(self, num_timesteps: int, beta_start: float, beta_end: float, batch_size: int, img_size: int, device: str):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.batch_size = batch_size
        self.img_size = img_size
        self.device = device

        # All these tensors have dimension --> [1000] if num_timesteps = 1000
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alphas = 1. - self.betas.to(device)
        self.alpha_bars = torch.cumprod(self.alphas, dim=0).to(device)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars).to(device)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1. - self.alpha_bars).to(device)

    def add_noise(self, x_0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor):
        # sqrt_alpha_bars[t] has shape [batch_size] and we need to multiply it with [batch_size, 1, 28, 28] image.
        # for broadcasting to work correctly, we are adding dimensions to make it [batch_size, 1, 1, 1]
        x_0 = x_0.to(self.device)
        sqrt_alpha_bars = self.sqrt_alpha_bars[t].reshape([-1, 1, 1, 1]).to(self.device)   # Bx1x1x1
        sqrt_one_minus_alpha_bars = self.sqrt_one_minus_alpha_bars[t].reshape([-1, 1, 1, 1]).to(self.device)   # Bx1x1x1

        # the direct formula to get x_t from x_0
        return sqrt_alpha_bars*x_0 + sqrt_one_minus_alpha_bars*noise
    
    def sampling(self, unet: torch.nn.Module, batch_size):
        x_t = torch.randn((batch_size, 1, self.img_size, self.img_size)).to(self.device)
        for t in range(self.num_timesteps-1, -1, -1):
            # t is an integer. Want make tensor that contains t with shape -> [batch_size]
            T = torch.tensor([t]).expand(batch_size).to(self.device)
            # sigma_1 = torch.sqrt(self.betas[T]) # fixed large variance (choice 1 in paper)
            sigma_2 = (self.sqrt_one_minus_alpha_bars[T-1]/self.sqrt_one_minus_alpha_bars[T]) * torch.sqrt(self.betas[T])   # tighter posterior variance (choice 2)
            noise = sigma_2.reshape(-1, 1, 1, 1) * torch.randn((batch_size, 1, self.img_size, self.img_size)).to(self.device)
            unet_pred_noise = unet(x_t, T)
            if t == 0:
                noise = torch.zeros_like(noise).to(self.device)
            x_t = (1/torch.sqrt(self.alphas[T])).reshape(-1, 1, 1, 1) * (x_t - ((self.betas[T]/self.sqrt_one_minus_alpha_bars[T]).reshape(-1, 1, 1, 1) * unet_pred_noise)) + noise
        return x_t