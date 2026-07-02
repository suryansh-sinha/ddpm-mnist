import matplotlib.pyplot as plt
import torch
from diffusion import LinearNoiseScheduler

def visualize_forward_diffusion(data_loader: torch.utils.data.DataLoader):
    """Selects a random image from the dataloader and then applies the forward diffusion process to it,
    adding noise to it in small steps till the image becomes pure gaussian noise.

    Args:
        data_loader (torch.utils.data.DataLoader): The DataLoader for our dataset
    """
    train_img, train_label = next(iter(data_loader))
    train_img.shape, train_label.shape

    test_img = train_img[0]
    test_img_batched = torch.stack([test_img for _ in range(6)], dim=0)
    print(test_img_batched.shape)
    noise = torch.randn_like(test_img_batched)
    t = [0, 100, 250, 500, 750, 999]

    lns = LinearNoiseScheduler(1000, 1e-4, 0.02)
    out_imgs = lns.add_noise(test_img_batched, noise=noise, t=t)
    out_imgs = (out_imgs + 1)/2
    print(out_imgs[1].shape)

    fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(15, 3))
    for i in range(6):
        img = out_imgs[i].squeeze()
        img_numpy = img.cpu().numpy()

        axes[i].imshow(img_numpy, cmap="gray")
        axes[i].set_title(f"Image {i+1}")
        axes[i].axis("off") # Hide the grid and the pixel coordinates
    plt.tight_layout()
    plt.show()