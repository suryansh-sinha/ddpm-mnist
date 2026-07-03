import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import v2
from src.unet import UNet
from src.diffusion import LinearNoiseScheduler

import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Loading our dataset
transform = v2.Compose([
    v2.ToImage(), 
    v2.ToDtype(torch.float32, scale=True),
    v2.Lambda(lambda x: x*2 - 1)
])
dataset = FashionMNIST(root="./data", train=True, transform=transform, download=True)


def save_model_checkpoint(path, epoch, model, ema_model, optimizer, losses):
    model_checkpoint = {
        "current_epoch": epoch,
        "model_state": model.state_dict(),
        "ema_model_state": ema_model.module.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "loss_history": losses,
    }
    torch.save(model_checkpoint, path)

# Hyperparameters
BATCH_SIZE = 128
LR = 1e-4   # lowered from 2e-4 because smaller image dimensions in FashionMNIST
EPOCHS = 10
NUM_TIMESTEPS = 1000
BETA_START = 1e-4
BETA_END = 0.02
IMG_SIZE = 28
IN_CHANNELS = 1
BASE_CHANNELS = 64  # paper used 128
TIME_EMB_DIM = 128
EMB_DIM = 256
N_HEADS = 8
NUM_GROUPS = 8
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# Objects
lns = LinearNoiseScheduler(NUM_TIMESTEPS, BETA_START, BETA_END, BATCH_SIZE, IMG_SIZE, DEVICE)
model = UNet(IN_CHANNELS, BASE_CHANNELS, TIME_EMB_DIM, EMB_DIM, N_HEADS, NUM_GROUPS).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr = LR)
ema_model = torch.optim.swa_utils.AveragedModel(
    model, 
    multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.9999)
)
criterion = torch.nn.MSELoss()

# Create DataLoader
train_loader = DataLoader(dataset, BATCH_SIZE, shuffle=True, num_workers=8)

# Training starts
losses = []

for epoch in range(EPOCHS):

    model.train()   # Set model to training mode
    train_loss = 0.0

    for batch_idx, (x_0, _) in enumerate(train_loader):
        optimizer.zero_grad()
        x_0 = x_0.to(DEVICE)

        # t ~ Uniform ({1...T})
        t = torch.randint(low=0, high=NUM_TIMESTEPS, size=(BATCH_SIZE, )).to(DEVICE)

        # noise ~ N(0, I)
        noise = torch.randn_like(x_0).to(DEVICE)

        # forward diffusion process
        x_t = lns.add_noise(x_0, noise, t)

        # Loss -> ||noise - unet_pred(add_noise function to timesteps t, timestep t itself)|| ** 2
        predicted_noise = ema_model.module(x_t, t) # getting model prediction
        loss = criterion(predicted_noise, noise)

        loss.backward()     # calculate gradients
        optimizer.step()    # update model parameters

        ema_model.update_parameters(model)

        if batch_idx % 100 == 0:
            print(f"Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.5f}")

        train_loss += loss.item() * x_0.size(0)
    
    epoch_train_loss = train_loss / (len(train_loader)*BATCH_SIZE)
    losses.append(epoch_train_loss)

    if epoch % 20 == 0:
        print(f"Epoch: {epoch}/{EPOCHS} | Loss: {epoch_train_loss:.5f}")

# Create the plot
# plt.figure(figsize=(16, 10))
# plt.plot(losses, label='Training Loss', color='blue', linewidth=2)
# plt.title("Loss Curve")
# plt.xlabel("epochs")
# plt.ylabel("loss")
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.show()

# testing the model
# model.eval()
# with torch.inference_mode():
#     outputs = lns.sampling(model, BATCH_SIZE)  
#     outputs = (outputs + 1) / 2
#     inputs = (x_0 + 1) / 2

#     fig, axes = plt.subplots(nrows=2, ncols=8, figsize=(15, 6))
    
#     for i in range(8):
#         # Row 0: Input Images
#         img_in = inputs[i].squeeze().cpu().numpy()
#         axes[0, i].imshow(img_in, cmap="gray")
#         axes[0, i].set_title(f"Input {i+1}")
#         axes[0, i].axis("off")

#         # Row 1: Generated Images
#         img_out = outputs[i].squeeze().cpu().numpy()
#         axes[1, i].imshow(img_out, cmap="gray")
#         axes[1, i].set_title(f"Generated {i+1}")
#         axes[1, i].axis("off")
        
#     plt.tight_layout()
#     plt.show()