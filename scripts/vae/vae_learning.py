import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import os
import matplotlib.pyplot as plt

# Define the VAE architecture
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        # Encoder
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU()
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Loss function for VAE
def vae_loss(reconstructed, original, mu, logvar, beta=0.1):
    reconstruction_loss = nn.MSELoss()(reconstructed, original)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss + beta * kl_divergence

# Load normalized data
def load_normalized_data(filepath):
    return pd.read_csv(filepath)

def generate_pose_sequences(df, keypoints, sequence_length=30):
    keypoint_columns = [f"{kp}_x" for kp in keypoints] + [f"{kp}_y" for kp in keypoints]
    data = df[keypoint_columns].values
    sequences = [data[i:i + sequence_length].flatten() for i in range(len(data) - sequence_length + 1)]
    return np.array(sequences)

if __name__ == "__main__":
    # Parameters
    latent_dim = 1000
    sequence_length = 30
    batch_size = 64
    epochs = 100
    save_path = "../../dataset/2D-poses/shadow/shadow_dataset_normalized.csv"

    # Load data
    normalized_df = load_normalized_data(save_path)
    selected_keypoints = ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 
                          'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee']
    sequences = generate_pose_sequences(normalized_df, selected_keypoints, sequence_length)

    # Split into train and test sets
    sequences = torch.tensor(sequences, dtype=torch.float32)
    dataset = TensorDataset(sequences)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize VAE
    input_dim = sequences.shape[1]
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize VAE and move it to the device
vae = VAE(input_dim, latent_dim).to(device)
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

# Training loop
for epoch in range(epochs):
    vae.train()
    train_loss = 0
    for batch in train_loader:
        batch = batch[0].to(device)  # Move batch to GPU
        optimizer.zero_grad()
        reconstructed, mu, logvar = vae(batch)
        loss = vae_loss(reconstructed, batch, mu, logvar)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss / len(train_loader)}")

# Save the model
torch.save(vae.state_dict(), "../../models/vae/shadow_boxing_vae.pth")
print("VAE model saved.")

# Evaluation on test set
vae.eval()
test_losses = []
with torch.no_grad():
    for batch in test_loader:
        batch = batch[0].to(device)  # Move batch to GPU
        reconstructed, mu, logvar = vae(batch)
        loss = vae_loss(reconstructed, batch, mu, logvar)
        test_losses.append(loss.item())

    
    # Histogram of reconstruction losses (as proxy for likelihood)
    plt.hist(test_losses, bins=30, alpha=0.7)
    plt.title("Reconstruction Loss Distribution (Test Set)")
    plt.xlabel("Loss")
    plt.ylabel("Frequency")
    plt.show()
