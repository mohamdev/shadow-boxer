import pandas as pd
import numpy as np
import cv2
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
import joblib  # For saving and loading the model
import os  # For directory management
import matplotlib.pyplot as plt
# Import your custom draw function
from rtmlib import draw_skeleton

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

def train_gmm(train_df, n_components=10, random_state=42):
    """
    Train a Gaussian Mixture Model (GMM) on the training dataset.
    """
    # Extract relevant data: drop non-keypoint columns (e.g., pose_score)
    keypoint_columns = [col for col in train_df.columns if '_x' in col or '_y' in col]
    train_data = train_df[keypoint_columns].values

    # Train the GMM
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    gmm.fit(train_data)
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

def evaluate_pose_likelihood(gmm, pose):
    """
    Evaluate the likelihood that a pose belongs to the shadow-boxing dataset.
    """
    pose = pose.reshape(1, -1)  # Reshape to match GMM input format
    log_likelihood = gmm.score(pose)
    return log_likelihood

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

    # Train the GMM
    gmm = train_gmm(train_df, n_components=30)

    # Save the trained GMM model
    model_save_path = "../../models/gmm/shadow_boxing_gmm.pkl"
    save_gmm_model(gmm, model_save_path)

    # Optionally load the saved model to verify
    loaded_gmm = load_gmm_model(model_save_path)

    # Evaluate the GMM on a random test pose
    test_pose = test_df.iloc[0]  # Select the first pose in the test set
    keypoint_columns = [col for col in test_df.columns if '_x' in col or '_y' in col]
    test_pose_values = test_pose[keypoint_columns].values

    likelihood = evaluate_pose_likelihood(loaded_gmm, test_pose_values)
    print(f"Log-Likelihood of the selected test pose: {likelihood}")

    # Optional: Evaluate likelihoods for all test poses
    test_data = test_df[keypoint_columns].values
    test_likelihoods = loaded_gmm.score_samples(test_data)

    plt.hist(test_likelihoods, bins=30, alpha=0.7)
    plt.title("Log-Likelihood Distribution")
    plt.xlabel("Log-Likelihood")
    plt.ylabel("Frequency")
    plt.show()

    print(f"Log-Likelihoods for all test poses: {test_likelihoods}")
