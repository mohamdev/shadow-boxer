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

def train_gmm_on_sequences(sequences, n_components=10, random_state=42):
    """
    Train a Gaussian Mixture Model (GMM) on the sequences dataset.

    Parameters:
    - sequences: numpy array of flattened pose sequences.
    - n_components: Number of Gaussian components in the GMM.
    - random_state: Random seed for reproducibility.

    Returns:
    - Trained GMM model.
    """
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    gmm.fit(sequences)
    return gmm

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

############################################
# Main script
############################################

if __name__ == "__main__":

    # Load the normalized data
    save_path = "../../dataset/2D-poses/shadow/shadow_dataset_normalized.csv"
    normalized_df = load_normalized_data(save_path)

    # Split the dataset into training and test sets
    train_df, test_df = split_data(normalized_df)

    print(f"Training set size: {len(train_df)}, Test set size: {len(test_df)}")

    # Define the keypoints to include in the GMM
    selected_keypoints = [
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee'
    ]
    
    # # Define the keypoints to include in the GMM
    # selected_keypoints = [
    #     'left_shoulder', 'right_shoulder',
    #     'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    #     'left_knee', 'right_knee'
    # ]

    # Generate pose sequences
    sequence_length = 30
    train_sequences = generate_pose_sequences(train_df, selected_keypoints, sequence_length=sequence_length)
    test_sequences = generate_pose_sequences(test_df, selected_keypoints, sequence_length=sequence_length)

    print(f"Training set sequences: {train_sequences.shape}")
    print(f"Test set sequences: {test_sequences.shape}")

    # Train the GMM
    gmm = train_gmm_on_sequences(train_sequences, n_components=50)

    # Save the trained GMM model
    model_save_path = "../../models/gmm/shadow_boxing_gmm_window.pkl"
    save_gmm_model(gmm, model_save_path)

    # Optionally load the saved model to verify
    loaded_gmm = load_gmm_model(model_save_path)

    # Evaluate the GMM on a random test sequence
    test_sequence = test_sequences[0]  # Select the first sequence in the test set
    likelihood = loaded_gmm.score(test_sequence.reshape(1, -1))
    print(f"Log-Likelihood of the selected test sequence: {likelihood}")

    # Optional: Evaluate likelihoods for all test sequences
    test_likelihoods = loaded_gmm.score_samples(test_sequences)

    plt.hist(test_likelihoods, bins=30, alpha=0.7)
    plt.title("Log-Likelihood Distribution for Sequences")
    plt.xlabel("Log-Likelihood")
    plt.ylabel("Frequency")
    plt.show()

    print(f"Log-Likelihoods for all test sequences: {test_likelihoods}")
