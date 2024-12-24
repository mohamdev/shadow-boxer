import cv2
import time
from rtmlib import RTMO, draw_skeleton
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.mixture import GaussianMixture

# Configuration
device = 'cuda'  # cpu, cuda, mps
backend = 'onnxruntime'  # opencv, onnxruntime, openvino

# Keypoint labels based on COCO 17 format
coco17_labels = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# Keypoints to keep for the VAE
relevant_keypoints = [
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]
relevant_indices = [coco17_labels.index(kp) for kp in relevant_keypoints]

# Load the VAE encoder and GMM model
vae_model_path = "../../models/vae_encoder.pth"
gmm_model_path = "../../models/gmm_latent.pkl"
latent_dim = 500

# Define the VAE class (must match training architecture)
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dims[1], latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        return mu

vae = VAE(input_dim=51 * 30, hidden_dims=[2048, 1024, 512] , latent_dim=latent_dim)
vae.load_state_dict(torch.load(vae_model_path, map_location=device))
vae.eval()
print(f"VAE encoder loaded from {vae_model_path}")

gmm = joblib.load(gmm_model_path)
print(f"GMM model loaded from {gmm_model_path}")

# Initialize the RTMPose model
pose_model = RTMO(
    onnx_model='../../models/pose_estimation/rtmo-l.onnx',  # Replace with your .onnx RTMO model
    model_input_size=(640, 640),
    backend=backend,
    device=device
)

def normalize_pose_np(filtered_keypoints):
    """
    Normalize a single pose (numpy array with relevant keypoints only) to be invariant to scale and translation,
    using the mean distance between left_shoulder to left_hip and right_shoulder to right_hip.
    """
    left_hip_idx = relevant_keypoints.index('left_hip')
    right_hip_idx = relevant_keypoints.index('right_hip')
    left_shoulder_idx = relevant_keypoints.index('left_shoulder')
    right_shoulder_idx = relevant_keypoints.index('right_shoulder')

    hip_center = (filtered_keypoints[left_hip_idx, :2] + filtered_keypoints[right_hip_idx, :2]) / 2
    filtered_keypoints[:, :2] -= hip_center

    left_dist = np.linalg.norm(filtered_keypoints[left_shoulder_idx, :2] - filtered_keypoints[left_hip_idx, :2])
    right_dist = np.linalg.norm(filtered_keypoints[right_shoulder_idx, :2] - filtered_keypoints[right_hip_idx, :2])
    scale = (left_dist + right_dist) * 2

    if scale > 0:
        filtered_keypoints[:, :2] /= scale

    return filtered_keypoints

def calculate_angle(v1, v2):
    """Calculate the angle between two vectors."""
    dot_product = np.dot(v1, v2)
    magnitude = np.linalg.norm(v1) * np.linalg.norm(v2)
    if magnitude == 0:
        return 0.0
    angle = np.arccos(np.clip(dot_product / magnitude, -1.0, 1.0))
    return angle / 100

def calculate_speeds(raw_keypoints_buffer):
    """Calculate normalized speed for each keypoint."""
    speeds = []
    for i in range(1, len(raw_keypoints_buffer)):
        frame_speed = []
        for j in range(raw_keypoints_buffer[i].shape[0]):
            dx = raw_keypoints_buffer[i][j, 0] - raw_keypoints_buffer[i - 1][j, 0]
            dy = raw_keypoints_buffer[i][j, 1] - raw_keypoints_buffer[i - 1][j, 1]
            speed = np.sqrt(dx**2 + dy**2) / 100
            frame_speed.append(speed)
        speeds.append(frame_speed)

    if len(speeds) == 0:
        speeds.append([0] * raw_keypoints_buffer[0].shape[0])  # No speed for the first frame

    return np.mean(speeds, axis=0)

def extract_features(raw_keypoints, scores, raw_keypoints_buffer):
    """Extract pose, angles, and speed features from raw keypoints."""
    pose_score = np.mean(scores)
    features = raw_keypoints.flatten().tolist() + [pose_score]

    left_shoulder = raw_keypoints[relevant_keypoints.index('left_shoulder')]
    right_shoulder = raw_keypoints[relevant_keypoints.index('right_shoulder')]
    left_elbow = raw_keypoints[relevant_keypoints.index('left_elbow')]
    right_elbow = raw_keypoints[relevant_keypoints.index('right_elbow')]
    left_wrist = raw_keypoints[relevant_keypoints.index('left_wrist')]
    right_wrist = raw_keypoints[relevant_keypoints.index('right_wrist')]

    left_upper_arm = left_elbow - left_shoulder
    left_lower_arm = left_wrist - left_elbow
    right_upper_arm = right_elbow - right_shoulder
    right_lower_arm = right_wrist - right_elbow

    left_arm_angle = calculate_angle(left_upper_arm, left_lower_arm)
    right_arm_angle = calculate_angle(right_upper_arm, right_lower_arm)

    features.extend([left_arm_angle, right_arm_angle])

    # Add speed features
    speed_features = calculate_speeds(raw_keypoints_buffer)
    features.extend(speed_features)

    return np.array(features)

output_filename = "../../dataset/videos/validation-videos/gmm_classification_output5.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_filename, fourcc, 30, (1280, 720))

# Open the video file or webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Error: Unable to access the video source.")
    exit()

print("Processing video... Press 'q' to stop.")

# Initialize pose buffer for the sliding window
buffer_size = 30
pose_buffer = []
raw_keypoints_buffer = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or unable to capture frame.")
        break

    # Perform pose estimation
    keypoints, scores = pose_model(frame)
    keypoints = keypoints[:1]
    scores = scores[:1]
    if keypoints is not None and scores is not None:
        for i in range(keypoints.shape[0]):  # Iterate over detected poses
            raw_keypoints = keypoints[i, relevant_indices, :2]

            raw_keypoints_buffer.append(raw_keypoints)
            if len(raw_keypoints_buffer) > buffer_size:
                raw_keypoints_buffer.pop(0)

            normalized_keypoints = normalize_pose_np(raw_keypoints.copy())

            features = extract_features(raw_keypoints, scores[i], raw_keypoints_buffer)
            pose_buffer.append(features)
            if len(pose_buffer) > buffer_size:
                pose_buffer.pop(0)

            if len(pose_buffer) == buffer_size:
                # Convert buffer to tensor and encode with VAE
                buffer_tensor = torch.tensor(np.concatenate(pose_buffer), dtype=torch.float32).unsqueeze(0)
                latent_vector = vae.encode(buffer_tensor.to(device)).cpu().detach().numpy()

                # Measure GMM inference time
                start_time = time.time()
                predicted_class = gmm.predict(latent_vector)[0]  # Predict class
                class_probabilities = gmm.predict_proba(latent_vector)[0]  # Get probabilities for each class
                gmm_inference_time = (time.time() - start_time) * 1000  # ms

                # Define class messages
                class_labels = ["NOT GUARDING", "GUARDING", "STRIKING!!!"]
                class_colors = [(255, 0, 0)] * len(class_labels)  # Default red for all classes
                class_colors[predicted_class] = (0, 255, 0)  # Green for predicted class

                # Overlay all class labels with their probabilities
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.9
                thickness = 2

                for idx, label in enumerate(class_labels):
                    class_text = f"{label}: {class_probabilities[idx]:.2f}"
                    text_size, _ = cv2.getTextSize(class_text, font, font_scale, thickness)
                    text_x = frame.shape[1] - text_size[0] - 10
                    text_y = 30 + idx * (text_size[1] + 10)
                    cv2.putText(frame, class_text, (text_x, text_y), font, font_scale, class_colors[idx], thickness)

                print(f"GMM Inference Time: {gmm_inference_time:.2f} ms, Predicted Class: {predicted_class}, Probabilities: {class_probabilities}")
            break

    # Display skeleton and frame
    frame = draw_skeleton(frame, keypoints, scores, kpt_thr=0.5)
    cv2.imshow('Pose Estimation and Classification', frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
out.release()
cap.release()
cv2.destroyAllWindows()
