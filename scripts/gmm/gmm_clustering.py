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

def split_data(normalized_df, test_size=0):
    """
    Split the dataset into learning and test sets by taking the first `1-test_size` for training
    and the remaining for testing.
    """
    split_index = int(len(normalized_df) * (1 - test_size))
    train_df = normalized_df.iloc[:split_index]
    test_df = normalized_df.iloc[split_index:]
    return train_df, test_df

def generate_pose_sequences(df, keypoints, sequence_length):
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
    gmm = GaussianMixture(
        n_components=n_components,
        random_state=random_state,
        verbose=2,
        verbose_interval=1
    )
    gmm.fit(sequences)
    cluster_labels = gmm.predict(sequences)
    return gmm, cluster_labels

def save_gmm_model(gmm, model_path):
    """
    Save the trained GMM model to a specified path.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(gmm, model_path)
    print(f"GMM model saved to {model_path}")

def analyze_clusters(cluster_labels, n_components):
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

    plt.bar(range(n_components), cluster_counts, alpha=0.7)
    plt.title("Cluster Distribution")
    plt.xlabel("Cluster")
    plt.ylabel("Number of Sequences")
    plt.show()

if __name__ == "__main__":
    save_path = "../../dataset/2D-poses/shadow/shadow_dataset_poses_normalized_newset.csv"
    print("Loading data...")
    normalized_df = load_normalized_data(save_path)

    print("Data loaded, splitting data...")
    train_df, test_df = split_data(normalized_df)

    print("Data split, generating pose sequences...")
    selected_keypoints = [
        'left_shoulder', 'right_shoulder', 
        'left_elbow', 'right_elbow', 'left_wrist', 
        'right_wrist', , 'left_hip', 'right_hip',
        'left_knee', 'right_knee' #, 'left_ankle', 'right_ankle'
    ]
    # selected_keypoints = [
    #     'left_elbow', 'right_elbow',
    #     'left_wrist', 'right_wrist', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    # ]
    sequence_sizes = [10]
    n_classes_list = [3]

    for sequence_size in sequence_sizes:
        train_sequences = generate_pose_sequences(train_df, selected_keypoints, sequence_length=sequence_size)

        for n_classes in n_classes_list:
            print(f"Training GMM for {n_classes} classes and sequence size {sequence_size}...")
            gmm, cluster_labels = train_gmm_with_clustering(train_sequences, n_components=n_classes)

            print(f"Analyzing clusters for {n_classes} classes and sequence size {sequence_size}...")
            analyze_clusters(cluster_labels, n_components=n_classes)

            model_save_path = f"../../models/gmm/shadow_boxing_gmm_{n_classes}classes_{sequence_size}seq_newset.pkl"
            save_gmm_model(gmm, model_save_path)
