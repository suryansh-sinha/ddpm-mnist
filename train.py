import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from src.unet import UNet
from src.load_data import TransistorDataset
from src.diffusion import LinearNoiseScheduler
from src.utils import save_model_checkpoint

import os
import wandb
from dotenv import load_dotenv
import matplotlib.pyplot as plt

load_dotenv()

# Loading our dataset
transform = v2.Compose([
    v2.RandomHorizontalFlip(),
    v2.Lambda(lambda x: x*2 - 1)
])
# dataset = FashionMNIST(root="./data", train=True, transform=transform, download=True)
dataset = TransistorDataset(img_dir="./data/transistor/train/good/", transform=transform)

# Hyperparameters
BATCH_SIZE = 4
LR = 2e-4   # paper default
EPOCHS = 10
NUM_TIMESTEPS = 1000
BETA_START = 1e-4
BETA_END = 0.02
IMG_SIZE = 1024
IN_CHANNELS = 3
BASE_CHANNELS = 16  # paper used 128
TIME_EMB_DIM = 128
EMB_DIM = 256
N_HEADS = 8
NUM_GROUPS = 8
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

config = {
    "BATCH_SIZE": BATCH_SIZE,
    "LR": LR,
    "EPOCHS": EPOCHS,
    "NUM_TIMESTEPS": NUM_TIMESTEPS,
    "BETA_START": BETA_START,
    "BETA_END": BETA_END,
    "IMG_SIZE": IMG_SIZE,
    "IN_CHANNELS": IN_CHANNELS,
    "BASE_CHANNELS": BASE_CHANNELS,
    "TIME_EMB_DIM": TIME_EMB_DIM,
    "EMB_DIM": EMB_DIM,
    "N_HEADS": N_HEADS,
    "NUM_GROUPS": NUM_GROUPS,
    "DEVICE": DEVICE,
}

wandb.login(key=os.getenv("WANDB_API_KEY"))
wandb.init(project='DDPM-MVTecAD', config=config)

# Objects
lns = LinearNoiseScheduler(NUM_TIMESTEPS, BETA_START, BETA_END, DEVICE)
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
fixed_noise = torch.randn((8, IN_CHANNELS, IMG_SIZE, IMG_SIZE), device=DEVICE, requires_grad=False) # Fixed noise to test generated samples
best_loss = 1000
losses = []

for epoch in range(EPOCHS):

    model.train()   # Set model to training mode
    train_loss = 0.0
    total_imgs = 0

    for batch_idx, x_0 in enumerate(train_loader):
        optimizer.zero_grad()
        x_0 = x_0.to(DEVICE)
        
        # t ~ Uniform ({1...T})
        t = torch.randint(low=0, high=NUM_TIMESTEPS, size=(x_0.size(0), )).to(DEVICE)

        # noise ~ N(0, I)
        noise = torch.randn_like(x_0).to(DEVICE)

        # forward diffusion process
        x_t = lns.add_noise(x_0, noise, t)

        # Loss -> ||noise - unet_pred(add_noise function to timesteps t, timestep t itself)|| ** 2
        predicted_noise = model(x_t, t) # getting model prediction
        loss = criterion(predicted_noise, noise)

        loss.backward()     # calculate gradients
        optimizer.step()    # update model parameters

        ema_model.update_parameters(model)

        if (batch_idx) % 6 == 0:
            print(f"Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.5f}")

        train_loss += loss.item() * x_0.size(0)
        wandb.log({"batch_loss": train_loss, "batch_idx": batch_idx})
        total_imgs += x_0.size(0)
    
    # In case the total dataset size is not divisible by batch_size, we explicitly count number of images in last batch and add them for correct division to get epoch_loss
    epoch_train_loss = train_loss / (total_imgs)
    wandb.log({"epoch_loss": epoch_train_loss, "epoch": epoch})

    # Saving best performing model
    if epoch_train_loss < best_loss:
        best_loss = epoch_train_loss
        save_model_checkpoint("./outputs/best.pt", epoch, model, ema_model, optimizer, losses)

    losses.append(epoch_train_loss)

    if (epoch+1) % 2 == 0:
        print(f"Epoch: {epoch+1}/{EPOCHS} | Loss: {epoch_train_loss:.5f}")
        save_model_checkpoint(f"./outputs/model_epoch_{epoch+1}.pt", epoch, model, ema_model, optimizer, losses)
        # sampling logic
        ema_model.eval()
        with torch.inference_mode():
            generated = lns.sampling(ema_model.module)
        # log generated samples to w&b

# Create the plot
# plt.figure(figsize=(16, 10))
# plt.plot(losses, label='Training Loss', color='blue', linewidth=2)
# plt.title("Loss Curve")
# plt.xlabel("epochs")
# plt.ylabel("loss")
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.show()