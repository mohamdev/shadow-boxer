import pandas as pd
import numpy as np
import cv2
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
import joblib
import os
import matplotlib.pyplot as plt

def load_normalized_data(filepath):
    """
    Load the normalized pose data from a CSV file.
    """
    return pd.read_csv(filepath)

def split_data(normalized_df, test_size=0.2, random_state=42):
    """
    Split the dataset into learning and test sets.
    """
    train_df, test_df = train_test_split(normalized_df, test_size=test_size, random_state=random_state)
    return train_df, test_df

def generate_pose_sequences(df, keypoints, sequence_length=30):
    """
    Generate sequences of consecutive poses and flatten them.

    Parameters:
    - df: DataFrame containing normalized pose data.
    - keypoints: List of keypoints to include in the sequences.
    - sequence_length: Number of poses per sequence.

    Returns:
    - numpy array of flattened sequences.
    """
    keypoint_columns = []
    for kp in keypoints:
        keypoint_columns.extend([f"{kp}_x", f"{kp}_y"])

    # Validate that the specified columns exist in the DataFrame
    missing_columns = [col for col in keypoint_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"The following keypoint columns are missing from the DataFrame: {missing_columns}")

    # Extract the relevant keypoints
    data = df[keypoint_columns].values

    # Create sequences using a sliding window
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        sequence = data[i:i + sequence_length].flatten()  # Flatten the sequence
        sequences.append(sequence)

    return np.array(sequences)

def generate_pose_sequences_all(df, sequence_length=30):
    """
    Generate sequences of consecutive poses and flatten them.

    Parameters:
    - df: DataFrame containing normalized pose data.
    - sequence_length: Number of poses per sequence.

    Returns:
    - numpy array of flattened sequences.
    """
    # Extract all features from the DataFrame
    data = df.values  # Includes all columns

    # Create sequences using a sliding window
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        sequence = data[i:i + sequence_length].flatten()  # Flatten the sequence
        sequences.append(sequence)

    return np.array(sequences)


def train_gmm_with_clustering(sequences, n_components=3, random_state=42):
    """
    Train a Gaussian Mixture Model (GMM) with clustering on the sequences dataset.

    Parameters:
    - sequences: numpy array of flattened pose sequences.
    - n_components: Number of Gaussian components (clusters) in the GMM.
    - random_state: Random seed for reproducibility.

    Returns:
    - Trained GMM model.
    - Cluster labels for each sequence.
    """
    print("GMM instanciation...")
    gmm = GaussianMixture(
        n_components=n_components,
        random_state=random_state,
        verbose=2,  # Level of verbosity
        verbose_interval=1  # Log output every 1 iterations
    )
    print("Instanciated, gmm fitting...")
    gmm.fit(sequences)
    print("fitted, gmm predicting...")
    cluster_labels = gmm.predict(sequences)  # Cluster assignment for each sequence
    print("gmm training done.")
    return gmm, cluster_labels

def save_gmm_model(gmm, model_path):
    """
    Save the trained GMM model to a specified path.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(gmm, model_path)
    print(f"GMM model saved to {model_path}")

def load_gmm_model(model_path):
    """
    Load a saved GMM model from a specified path.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found at {model_path}")
    gmm = joblib.load(model_path)
    print(f"GMM model loaded from {model_path}")
    return gmm

def analyze_clusters(cluster_labels, n_components=3):
    """
    Analyze the distribution of sequences across clusters.

    Parameters:
    - cluster_labels: Cluster assignments for each sequence.
    - n_components: Number of clusters.

    Returns:
    - None
    """
    cluster_counts = np.bincount(cluster_labels)
    print(f"Cluster distribution: {dict(zip(range(n_components), cluster_counts))}")

    # Visualize cluster distribution
    plt.bar(range(n_components), cluster_counts, alpha=0.7)
    plt.title("Cluster Distribution")
    plt.xlabel("Cluster")
    plt.ylabel("Number of Sequences")
    plt.show()


############################################
# Main script (modified for clustering)
############################################

if __name__ == "__main__":
    # Load the normalized data
    save_path = "../../dataset/2D-poses/shadow/shadow_dataset_normalized.csv"
    print("loading data...")
    normalized_df = load_normalized_data(save_path)

    print("data loaded, splitting data...")
    # Split the dataset into training and test sets
    train_df, test_df = split_data(normalized_df)
    print("data splitted, generating pose sequences...")
    # Define the keypoints to include in the GMM
    selected_keypoints = [
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]

    # Generate pose sequences
    sequence_length = 30
    # train_sequences = generate_pose_sequences(train_df, selected_keypoints, sequence_length=sequence_length)
    train_sequences = generate_pose_sequences_all(train_df, sequence_length=sequence_length)
    print("Pose sequences generated, training...")
    # Train the GMM and get cluster labels
    n_classes = 3
    gmm, cluster_labels = train_gmm_with_clustering(train_sequences, n_components=n_classes)

    print("analyzing clusters...")
    # Analyze clusters
    analyze_clusters(cluster_labels, n_components=n_classes)
    print("clusters abakysed, saving model...")
    # Save the trained GMM model
    model_save_path = "../../models/gmm/shadow_boxing_gmm_classes_allfeatures.pkl"
    save_gmm_model(gmm, model_save_path)
    print("model saved, saving cluster labels")
    # Save cluster labels for further analysis
    cluster_save_path = "../../models/gmm/shadow_boxing_cluster_labels_allfeatures.npy"
    np.save(cluster_save_path, cluster_labels)
    print(f"Cluster labels saved to {cluster_save_path}, performing PCA...")

    # Optionally visualize the sequences in a reduced dimension space
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(train_sequences)
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
    plt.title("PCA Visualization of Clusters")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()