import cv2
import time
import torch
import numpy as np
import os
from rtmlib import RTMO, draw_skeleton

# Avoid OpenMP runtime errors
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    'left_knee', 'right_knee'
]

relevant_indices = [coco17_labels.index(kp) for kp in relevant_keypoints]

# Define the VAE model
class VAE(torch.nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU()
        )
        self.fc_mu = torch.nn.Linear(128, latent_dim)
        self.fc_logvar = torch.nn.Linear(128, latent_dim)

        # Decoder
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, input_dim),
            torch.nn.Sigmoid()
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

# Load the trained VAE model
latent_dim = 1000
input_dim = len(relevant_keypoints) * 2 * 30  # 30-frame sliding window
vae = VAE(input_dim=input_dim, latent_dim=latent_dim).to(device)
vae.load_state_dict(torch.load("../../models/vae/shadow_boxing_vae.pth"))
vae.eval()
print("VAE model loaded and ready.")

# Initialize the RTMPose model
pose_model = RTMO(
    onnx_model='../../models/pose_estimation/rtmo-l.onnx',
    model_input_size=(640, 640),
    backend='onnxruntime',
    device='cuda'
)

def normalize_pose_np(filtered_keypoints):
    left_hip_idx = relevant_keypoints.index('left_hip')
    right_hip_idx = relevant_keypoints.index('right_hip')
    left_shoulder_idx = relevant_keypoints.index('left_shoulder')
    right_shoulder_idx = relevant_keypoints.index('right_shoulder')

    hip_center = (filtered_keypoints[left_hip_idx, :2] + filtered_keypoints[right_hip_idx, :2]) / 2
    filtered_keypoints[:, :2] -= hip_center

    left_dist = np.linalg.norm(filtered_keypoints[left_shoulder_idx, :2] - filtered_keypoints[left_hip_idx, :2])
    right_dist = np.linalg.norm(filtered_keypoints[right_shoulder_idx, :2] - filtered_keypoints[right_hip_idx, :2])
    scale = (left_dist + right_dist) / 2

    if scale > 0:
        filtered_keypoints[:, :2] /= scale

    return filtered_keypoints

trial_no = 5
output_filename = f"../../dataset/videos/validation-videos/vae_result_output{trial_no}.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_filename, fourcc, 30, (1920, 1080))

cap = cv2.VideoCapture(f"../../dataset/videos/validation-videos/output{trial_no}.mp4")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

if not cap.isOpened():
    print("Error: Unable to access the video source.")
    exit()

print("Processing video... Press 'q' to stop.")

buffer_size = 30
pose_buffer = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or unable to capture frame.")
        break

    keypoints, scores = pose_model(frame)

    log_likelihood_str = "Reconstruction Error: N/A"
    likelihood_color = (255, 0, 0)  # Blue

    if keypoints is not None and scores is not None:
        for i in range(keypoints.shape[0]):
            filtered_keypoints = keypoints[i, relevant_indices, :2]
            normalized_keypoints = normalize_pose_np(filtered_keypoints)

            pose_buffer.append(normalized_keypoints.flatten())
            if len(pose_buffer) > buffer_size:
                pose_buffer.pop(0)

            if len(pose_buffer) == buffer_size:
                buffer_vector = np.concatenate(pose_buffer)
                buffer_tensor = torch.tensor(buffer_vector, dtype=torch.float32).to(device)

                with torch.no_grad():
                    reconstructed, _, _ = vae(buffer_tensor.unsqueeze(0))
                    reconstruction_error = torch.nn.functional.mse_loss(reconstructed, buffer_tensor.unsqueeze(0)).item()

                log_likelihood_str = f"Reconstruction Error: {reconstruction_error:.4f}"
                likelihood_color = (0, 255, 0) if reconstruction_error < 0.05 else (0, 0, 255)
            break

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    thickness = 2

    text_size, _ = cv2.getTextSize(log_likelihood_str, font, font_scale, thickness)
    text_x = frame.shape[1] - text_size[0] - 10
    text_y = 30
    cv2.putText(frame, log_likelihood_str, (text_x, text_y), font, font_scale, likelihood_color, thickness)

    frame = draw_skeleton(frame, keypoints, scores, kpt_thr=0.8)
    cv2.imshow('Pose Estimation and Reconstruction Error', frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()
