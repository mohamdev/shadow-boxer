import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
import joblib
import os
import matplotlib.pyplot as plt

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super(VAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(hidden_dims[1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[1], latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], input_dim),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, logvar

def vae_loss_function(reconstructed, original, mu, logvar):
    reconstruction_loss = nn.MSELoss()(reconstructed, original)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss + 0.2*kl_divergence

def train_vae(vae, dataloader, epochs, learning_rate, device):
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
    vae.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x = batch[0].to(device)  # Move data to device
            optimizer.zero_grad()
            reconstructed, mu, logvar = vae(x)
            loss = vae_loss_function(reconstructed, x, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

def encode_data(vae, dataloader, device):
    vae.eval()
    latent_vectors = []
    with torch.no_grad():
        for batch in dataloader:
            x = batch[0].to(device)  # Move data to device
            h = vae.encoder(x)
            mu = vae.fc_mu(h)
            latent_vectors.append(mu.cpu().numpy())  # Move result to CPU
    return np.vstack(latent_vectors)

# Data loading and preprocessing functions

def load_normalized_data(filepath):
    return pd.read_csv(filepath)

def generate_pose_sequences_all(df, sequence_length=30):
    data = df.values
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        sequence = data[i:i + sequence_length].flatten()
        sequences.append(sequence)
    return np.array(sequences)

# Main script
if __name__ == "__main__":
    # Parameters
    data_path = "../../dataset/2D-poses/shadow/shadow_dataset_normalized.csv"
    model_save_path = "../../models/vae_encoder.pth"
    gmm_save_path = "../../models/gmm_latent.pkl"
    sequence_length = 30
    batch_size = 64
    learning_rate = 1e-3
    epochs = 20
    latent_dim = 1024  # Adjusted for larger latent space
    hidden_dims = [1024, 512, 512]  # Adjusted to support larger input and efficient encoding
    n_classes = 3

    # Device configuration
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    normalized_df = load_normalized_data(data_path)
    sequences = generate_pose_sequences_all(normalized_df, sequence_length=sequence_length)

    # Convert to PyTorch DataLoader
    sequences_tensor = torch.tensor(sequences, dtype=torch.float32)
    dataloader = DataLoader(TensorDataset(sequences_tensor), batch_size=batch_size, shuffle=True)

    # Initialize and train VAE
    input_dim = sequences.shape[1]
    print("input dim:", input_dim)
    # vae = VAE(input_dim=input_dim, hidden_dims=hidden_dims, latent_dim=latent_dim).to(device)  # Move model to device
    # train_vae(vae, dataloader, epochs, learning_rate, device)

    # # Save the trained encoder
    # torch.save(vae.state_dict(), model_save_path)
    # print(f"VAE encoder saved to {model_save_path}")

    # # Encode data to latent space
    # latent_vectors = encode_data(vae, dataloader, device)

    # # Train GMM on latent vectors
    # gmm = GaussianMixture(n_components=n_classes, random_state=42, verbose=2)
    # gmm.fit(latent_vectors)
    # joblib.dump(gmm, gmm_save_path)
    # print(f"GMM model saved to {gmm_save_path}")

    # # Optionally visualize clusters using PCA
    # from sklearn.decomposition import PCA
    # cluster_labels = gmm.predict(latent_vectors)
    # pca = PCA(n_components=2)
    # reduced_data = pca.fit_transform(latent_vectors)
    # plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
    # plt.title("PCA Visualization of Clusters")
    # plt.xlabel("Principal Component 1")
    # plt.ylabel("Principal Component 2")
    # plt.show()
