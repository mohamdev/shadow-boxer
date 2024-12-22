import cv2
import time
from rtmlib import RTMO, draw_skeleton
import joblib
import numpy as np

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

# Keypoints to keep for the GMM
# relevant_keypoints = [
#     'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
#     'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
#     'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
# ]

relevant_keypoints = [
    'left_shoulder', 'right_shoulder', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee']

relevant_indices = [coco17_labels.index(kp) for kp in relevant_keypoints]

# Load the GMM model
gmm_model_path = "../../models/gmm/shadow_boxing_gmm.pkl"
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
    # Indices for relevant keypoints in the filtered array
    left_hip_idx = relevant_keypoints.index('left_hip')
    right_hip_idx = relevant_keypoints.index('right_hip')
    left_shoulder_idx = relevant_keypoints.index('left_shoulder')
    right_shoulder_idx = relevant_keypoints.index('right_shoulder')

    # Calculate hip center
    hip_center = (filtered_keypoints[left_hip_idx, :2] + filtered_keypoints[right_hip_idx, :2]) / 2

    # Translate keypoints
    filtered_keypoints[:, :2] -= hip_center

    # Calculate scale
    left_dist = np.linalg.norm(filtered_keypoints[left_shoulder_idx, :2] - filtered_keypoints[left_hip_idx, :2])
    right_dist = np.linalg.norm(filtered_keypoints[right_shoulder_idx, :2] - filtered_keypoints[right_hip_idx, :2])
    scale = (left_dist + right_dist) / 2

    # Avoid division by zero
    if scale > 0:
        filtered_keypoints[:, :2] /= scale

    return filtered_keypoints

# Open the video file or webcam
trial_no = "6"
video_path = "../../dataset/videos/shadow/shadow" + trial_no + ".mp4"
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Error: Unable to access the video source.")
    exit()

print("Processing video... Press 'q' to stop.")

# Process the video stream
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or unable to capture frame.")
        break

    # Measure inference start time
    start_time = time.time()

    # Perform pose estimation
    keypoints, scores = pose_model(frame)

    # Measure inference end time
    end_time = time.time()
    inference_time = (end_time - start_time) * 1000  # Convert to milliseconds

    # Print resolution and inference time
    resolution = (frame.shape[1], frame.shape[0])  # (width, height)
    print(f"Resolution: {resolution[0]}x{resolution[1]} | Inference Time: {inference_time:.2f} ms")

    # Initialize the log-likelihood string
    log_likelihood_str = "Log-Likelihood: N/A"


    start_time = time.time()
    if keypoints is not None and scores is not None:
        for i in range(keypoints.shape[0]):  # Iterate over detected poses
            # Filter relevant keypoints
            filtered_keypoints = keypoints[i, relevant_indices, :2]

            # Normalize the pose
            normalized_keypoints = normalize_pose_np(filtered_keypoints)

            # Flatten the normalized keypoints
            pose_vector = normalized_keypoints.flatten()

            # Calculate log-likelihood
            log_likelihood = gmm.score(pose_vector.reshape(1, -1))
            log_likelihood_str = f"Log-Likelihood: {log_likelihood:.2f}"
            break  # Only consider the first pose for this example

    
    # Measure inference end time
    end_time = time.time()
    inference_time_likelihood = (end_time - start_time) * 1000  # Convert to milliseconds

    print(f"inference_time_likelihood: {inference_time_likelihood:.4f} ms")
    # Overlay the log-likelihood on the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_color = (0, 255, 0)  # Green
    thickness = 2
    text_size, _ = cv2.getTextSize(log_likelihood_str, font, font_scale, thickness)
    text_x = resolution[0] - text_size[0] - 10  # Top-right corner
    text_y = 30  # Slightly below the top edge

    cv2.putText(frame, log_likelihood_str, (text_x, text_y), font, font_scale, font_color, thickness)

    # Visualize the results
    frame = draw_skeleton(frame, keypoints, scores, kpt_thr=0.5)

    # Display the frame
    cv2.imshow('Pose Estimation and Log-Likelihood', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video source and close windows
cap.release()
cv2.destroyAllWindows()
