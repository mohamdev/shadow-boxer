import cv2
import time
from rtmlib import RTMO, draw_skeleton
import joblib
import numpy as np
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

# Keypoints to keep for classification
relevant_keypoints = [
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]
relevant_indices = [coco17_labels.index(kp) for kp in relevant_keypoints]

# Load GMM model
gmm_model_path = "../../models/gmm/shadow_boxing_gmm_classes_allfeatures.pkl"
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
    """Calculate normalized speed for each keypoint and return as a flat array."""
    if len(raw_keypoints_buffer) < 2:
        # No speed can be calculated for the first frame, so return zeros for all components
        return np.zeros(len(relevant_keypoints) * 2)

    speeds = []
    for idx, keypoint in enumerate(relevant_keypoints):
        dx = raw_keypoints_buffer[-1][idx, 0] - raw_keypoints_buffer[-2][idx, 0]
        dy = raw_keypoints_buffer[-1][idx, 1] - raw_keypoints_buffer[-2][idx, 1]
        norm = np.sqrt(dx**2 + dy**2)
        speeds.extend([norm / 100, norm / 100])  # Normalize by dividing by 100, matching training

    return np.array(speeds)

def extract_features(raw_keypoints, normalized_keypoints, scores, raw_keypoints_buffer):
    """Extract pose, angles, and speed features from normalized keypoints."""
    # Normalized pose keypoints
    pose_features = normalized_keypoints.flatten()

    # Pose score
    pose_score = np.array([np.mean(scores)])

    # Arm angles (using normalized keypoints)
    left_shoulder = normalized_keypoints[relevant_keypoints.index('left_shoulder')]
    right_shoulder = normalized_keypoints[relevant_keypoints.index('right_shoulder')]
    left_elbow = normalized_keypoints[relevant_keypoints.index('left_elbow')]
    right_elbow = normalized_keypoints[relevant_keypoints.index('right_elbow')]
    left_wrist = normalized_keypoints[relevant_keypoints.index('left_wrist')]
    right_wrist = normalized_keypoints[relevant_keypoints.index('right_wrist')]

    left_upper_arm = left_elbow - left_shoulder
    left_lower_arm = left_wrist - left_elbow
    right_upper_arm = right_elbow - right_shoulder
    right_lower_arm = right_wrist - right_elbow

    left_arm_angle = calculate_angle(left_upper_arm, left_lower_arm)
    right_arm_angle = calculate_angle(right_upper_arm, right_lower_arm)
    angles = np.array([left_arm_angle, right_arm_angle])

    # Speed keypoints (using raw keypoints)
    speed_features = calculate_speeds(raw_keypoints_buffer)

    # Concatenate features in the required order
    features = np.concatenate([pose_features, pose_score, angles, speed_features])

    # Ensure feature length matches the expected size (51 features)
    assert len(features) == 51, f"Expected 51 features, got {len(features)}"

    return features

output_filename = "../../dataset/videos/validation-videos/gmm_classification_output6.mp4"
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
        for i in range(keypoints.shape[0]):
            raw_keypoints = keypoints[i, relevant_indices, :2]

            raw_keypoints_buffer.append(raw_keypoints)
            if len(raw_keypoints_buffer) > buffer_size:
                raw_keypoints_buffer.pop(0)

            normalized_keypoints = normalize_pose_np(raw_keypoints.copy())
            features = extract_features(raw_keypoints, normalized_keypoints, scores[i], raw_keypoints_buffer)
            print("features vector:", features)
            pose_buffer.append(features)
            if len(pose_buffer) > buffer_size:
                pose_buffer.pop(0)

            if len(pose_buffer) == buffer_size:
                # Convert features buffer to numpy array
                feature_array = np.array(pose_buffer).flatten().reshape(1, -1)

                # Measure GMM inference time
                start_time = time.time()
                predicted_class = gmm.predict(feature_array)[0]  # Predict class
                class_probabilities = gmm.predict_proba(feature_array)[0]  # Get probabilities for each class
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
