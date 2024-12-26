import pandas as pd
import numpy as np
import os
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib

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
    """
    keypoint_columns = []
    for kp in keypoints:
        keypoint_columns.extend([f"{kp}_x", f"{kp}_y"])

    missing_columns = [col for col in keypoint_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"The following keypoint columns are missing from the DataFrame: {missing_columns}")

    data = df[keypoint_columns].values
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        sequence = data[i:i + sequence_length].flatten()
        sequences.append(sequence)

    return np.array(sequences)


def dbscan_clustering(sequences, min_cluster_size_ratio=0.01, eps=0.5):
    """
    Perform DBSCAN clustering on the sequences.

    Parameters:
    - sequences: Numpy array of sequences.
    - min_cluster_size_ratio: Minimum size of a cluster as a fraction of the dataset size.
    - eps: The maximum distance between two samples for one to be considered in the neighborhood of the other.

    Returns:
    - Cluster labels for each sequence.
    """
    min_samples = int(len(sequences) * min_cluster_size_ratio)
    dbscan = DBSCAN(eps=eps, min_samples=5)
    labels = dbscan.fit_predict(sequences)
    return labels


def save_clusters_to_csv(sequences, labels, output_dir):
    """
    Save each cluster to a separate CSV file, including noise points (-1 label).
    """
    os.makedirs(output_dir, exist_ok=True)
    unique_labels = np.unique(labels)

    for label in unique_labels:
        cluster_data = sequences[labels == label]
        cluster_df = pd.DataFrame(cluster_data)
        filename = os.path.join(output_dir, f"cluster_{label}.csv")
        cluster_df.to_csv(filename, index=False)
        print(f"Saved cluster {label} to {filename}")


def train_gmm_with_clusters(sequences, labels, n_components=3, random_state=42):
    """
    Train a Gaussian Mixture Model (GMM) on sequences based on DBSCAN clusters.
    """
    gmm = GaussianMixture(
        n_components=n_components,
        random_state=random_state,
        verbose=2,
        verbose_interval=1
    )
    gmm.fit(sequences[labels != -1])  # Exclude noise points for GMM training
    return gmm


def analyze_clusters(labels):
    """
    Display cluster distribution.
    """
    unique, counts = np.unique(labels, return_counts=True)
    cluster_distribution = dict(zip(unique, counts))
    print(f"Cluster distribution: {cluster_distribution}")

    plt.bar(unique, counts, alpha=0.7)
    plt.title("Cluster Distribution")
    plt.xlabel("Cluster Label")
    plt.ylabel("Number of Sequences")
    plt.show()


if __name__ == "__main__":
    save_path = "../../dataset/2D-poses/shadow/shadow_dataset_poses_normalized_newset.csv"
    print("Loading data...")
    normalized_df = load_normalized_data(save_path)[2200000:]

    print("Data loaded, splitting data...")
    train_df, _ = split_data(normalized_df)

    print("Data split, generating pose sequences...")
    selected_keypoints = [
        'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow', 'left_wrist',
        'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee'
    ]
    sequence_length = 10
    train_sequences = generate_pose_sequences(train_df, selected_keypoints, sequence_length=sequence_length)

    print("Performing DBSCAN clustering...")
    dbscan_labels = dbscan_clustering(train_sequences, eps=10)

    print("Analyzing DBSCAN clusters...")
    analyze_clusters(dbscan_labels)

    print("Saving clusters to CSV...")
    output_dir = "../../clusters/"
    save_clusters_to_csv(train_sequences, dbscan_labels, output_dir)

    print("Training GMM on DBSCAN clusters...")
    gmm = train_gmm_with_clusters(train_sequences, dbscan_labels, n_components=3)

    model_save_path = "../../models/gmm/shadow_boxing_gmm_trained_on_dbscan.pkl"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(gmm, model_save_path)
    print(f"GMM model saved to {model_save_path}")
