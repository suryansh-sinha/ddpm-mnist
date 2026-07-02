import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import v2
from src.unet import UNet
from src.diffusion import LinearNoiseScheduler

import numpy as np
import matplotlib.pyplot as plt

# Loading our dataset
transform = v2.Compose([
    v2.ToImage(), 
    v2.ToDtype(torch.float32, scale=True),
    v2.Lambda(lambda x: x*2 - 1)
])
dataset = FashionMNIST(root="./data", train=True, transform=transform, download=True)

# Hyperparameters
BATCH_SIZE = 8
LR = 2e-4
EPOCHS = 3000
NUM_TIMESTEPS = 2000
BETA_START = 1e-4
BETA_END = 0.02
IMG_SIZE = 28
IN_CHANNELS = 1
BASE_CHANNELS = 64
TIME_EMB_DIM = 128
EMB_DIM = 256
N_HEADS = 8
NUM_GROUPS = 8
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# Objects
lns = LinearNoiseScheduler(NUM_TIMESTEPS, BETA_START, BETA_END, BATCH_SIZE, IMG_SIZE, DEVICE)
model = UNet(IN_CHANNELS, BASE_CHANNELS, TIME_EMB_DIM, EMB_DIM, N_HEADS, NUM_GROUPS).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr = LR) 
criterion = torch.nn.MSELoss()

# Create DataLoader
train_loader = DataLoader(dataset, BATCH_SIZE, shuffle=True, num_workers=8)

# img, label = next(iter(train_loader))
# print(img.shape, label.shape)

# Training starts
model.train()   # Set model to training mode

losses = []
x_0, _ = next(iter(train_loader))

for epoch in range(EPOCHS):
    
    optimizer.zero_grad()
    x_0 = x_0.to(DEVICE)
    # print(x_0.shape, x_0.device)

    # t ~ Uniform ({1...T})
    t = torch.randint(low=0, high=NUM_TIMESTEPS, size=(BATCH_SIZE, )).to(DEVICE)
    # print(t.shape, t.device)

    # noise ~ N(0, I)
    noise = torch.randn_like(x_0).to(DEVICE)
    # print(noise.shape, noise.device)

    # forward diffusion process
    x_t = lns.add_noise(x_0, noise, t)
    # print(x_t.shape, x_t.device)

    # Loss -> ||noise - unet_pred(add_noise function to timesteps t, timestep t itself)|| ** 2
    predicted_noise = model(x_t, t) # getting model prediction
    # print(predicted_noise.shape, predicted_noise.device)
    loss = criterion(predicted_noise, noise)
    losses.append(loss.item())

    if epoch % 5 == 0:
        print(f"EPOCH: {epoch}/{EPOCHS} | Loss: {loss.item():.5f}")

    loss.backward()     # calculate gradients
    optimizer.step()    # update model parameters

# Create the plot
# plt.figure(figsize=(16, 10))
# plt.plot(losses, label='Training Loss', color='blue', linewidth=2)
# plt.title("Loss Curve")
# plt.xlabel("epochs")
# plt.ylabel("loss")
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.show()

# testing the model
model.eval()
with torch.inference_mode():
    outputs = lns.sampling(model, BATCH_SIZE)  
    outputs = (outputs + 1) / 2
    inputs = (x_0 + 1) / 2

    fig, axes = plt.subplots(nrows=2, ncols=8, figsize=(15, 6))
    
    for i in range(8):
        # Row 0: Input Images
        img_in = inputs[i].squeeze().cpu().numpy()
        axes[0, i].imshow(img_in, cmap="gray")
        axes[0, i].set_title(f"Input {i+1}")
        axes[0, i].axis("off")

        # Row 1: Generated Images
        img_out = outputs[i].squeeze().cpu().numpy()
        axes[1, i].imshow(img_out, cmap="gray")
        axes[1, i].set_title(f"Generated {i+1}")
        axes[1, i].axis("off")
        
    plt.tight_layout()
    plt.show()