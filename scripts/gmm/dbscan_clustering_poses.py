import pandas as pd
import numpy as np
import os
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import hdbscan


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

def apply_pca(poses, n_components=10):
    """
    Apply PCA to reduce the dimensionality of the poses.
    
    Parameters:
    - poses: Original dataset as a numpy array.
    - n_components: Number of principal components to keep.
    
    Returns:
    - Reduced dataset and PCA model.
    """
    pca = PCA(n_components=n_components)
    reduced_poses = pca.fit_transform(poses)
    total_variance_retained = np.sum(pca.explained_variance_ratio_)
    print(f"Total variance retained by PCA: {total_variance_retained:.2f}")
    return reduced_poses, pca

def dbscan_clustering(poses, min_samples=50, eps=1):
    """
    Perform DBSCAN clustering on the poses.

    Parameters:
    - poses: Numpy array of pose data.
    - min_samples: Minimum number of samples in a neighborhood to form a cluster.
    - eps: The maximum distance between two samples for them to be considered in the same neighborhood.

    Returns:
    - Cluster labels for each pose.
    """
    print(f"DBSCAN parameters: eps={eps}, min_samples={min_samples}")

    clusterer = hdbscan.HDBSCAN(min_cluster_size=100, min_samples=500, cluster_selection_epsilon=0.0)
    labels = clusterer.fit_predict(poses)
    # dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    # labels = dbscan.fit_predict(poses)
    print(f"DBSCAN completed. Found {len(set(labels)) - (1 if -1 in labels else 0)} clusters and {np.sum(labels == -1)} noise points.")
    return labels

def save_clusters_to_csv(original_poses, labels, output_dir, selected_keypoints):
    """
    Save each cluster to a separate CSV file, including noise points (-1 label).
    
    Parameters:
    - original_poses: Original dataset before PCA.
    - labels: Cluster labels for each pose.
    - output_dir: Directory to save the cluster CSV files.
    - selected_keypoints: List of keypoint names.
    """
    os.makedirs(output_dir, exist_ok=True)
    unique_labels = np.unique(labels)

    # Reconstruct the header order
    ordered_columns = []
    for keypoint in selected_keypoints:
        ordered_columns.append(f"{keypoint}_x")
        ordered_columns.append(f"{keypoint}_y")

    # print("ordered columns:", ordered_columns)
    # Convert the numpy array back into a DataFrame
    original_df = pd.DataFrame(original_poses, columns=ordered_columns)

    # Add cluster labels as a new column
    original_df["cluster_label"] = labels

    for label in unique_labels:
        cluster_df = original_df[original_df["cluster_label"] == label]

        filename = os.path.join(output_dir, f"cluster_{label}.csv")
        cluster_df.to_csv(filename, index=False)
        print(f"Saved cluster {label} to {filename}")


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
    plt.ylabel("Number of Poses")
    plt.show()

if __name__ == "__main__":
    save_path = "../../dataset/2D-poses/shadow/shadow_dataset_poses_normalized_newset.csv"
    # save_path = "../../clusters/guarding/cluster_1.csv"
    print("Loading data...")
    normalized_df = load_normalized_data(save_path)
    print("data size:", len(normalized_df))
    print("Data loaded, splitting data...")
    train_df, _ = split_data(normalized_df)

    print("Data split, preparing poses...")
    selected_keypoints = [
        'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow', 'left_wrist',
        'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee'
    ]
    keypoint_columns = [f"{kp}_{axis}" for kp in selected_keypoints for axis in ["x", "y"]]
    # print("keypoint columns:", keypoint_columns)
    poses = train_df[keypoint_columns].values

    print("Applying PCA to reduce dimensionality...")
    reduced_poses, pca = apply_pca(poses, n_components=10)

    print("Performing DBSCAN clustering...")
    dbscan_labels = dbscan_clustering(poses)

    print("Analyzing DBSCAN clusters...")
    analyze_clusters(dbscan_labels)

    print("Saving clusters to CSV...")
    output_dir = "../../clusters/"

    selected_keypoints = [
        'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow', 'left_wrist',
        'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    keypoint_columns = [f"{kp}_{axis}" for kp in selected_keypoints for axis in ["x", "y"]]
    poses = train_df[keypoint_columns].values
    save_clusters_to_csv(poses, dbscan_labels, output_dir, selected_keypoints)

    print("All done!")
