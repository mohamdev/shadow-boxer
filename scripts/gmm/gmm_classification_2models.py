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
relevant_keypoints_normalization = [
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]
relevant_indices_normalization = [coco17_labels.index(kp) for kp in relevant_keypoints_normalization]

relevant_keypoints = [
    'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]
relevant_indices = [coco17_labels.index(kp) for kp in relevant_keypoints]

# Initialize pose buffer for the sliding window
buffer_size = 10
n_classes = 3

# Load the GMM model
gmm_strike_model_path = f"../../models/gmm/shadow_boxing_gmm_2classes_10seq.pkl"
gmm_strike = joblib.load(gmm_strike_model_path)
print(f"GMM model loaded from {gmm_strike_model_path}")

gmm_guard_model_path = f"../../models/gmm/shadow_boxing_gmm_3classes_10seq.pkl"
gmm_guard = joblib.load(gmm_guard_model_path)
print(f"GMM model loaded from {gmm_guard_model_path}")

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
    left_hip_idx = relevant_keypoints_normalization.index('left_hip')
    right_hip_idx = relevant_keypoints_normalization.index('right_hip')
    left_shoulder_idx = relevant_keypoints_normalization.index('left_shoulder')
    right_shoulder_idx = relevant_keypoints_normalization.index('right_shoulder')

    hip_center = (filtered_keypoints[left_hip_idx, :2] + filtered_keypoints[right_hip_idx, :2]) / 2
    filtered_keypoints[:, :2] -= hip_center

    left_dist = np.linalg.norm(filtered_keypoints[left_shoulder_idx, :2] - filtered_keypoints[left_hip_idx, :2])
    right_dist = np.linalg.norm(filtered_keypoints[right_shoulder_idx, :2] - filtered_keypoints[right_hip_idx, :2])
    scale = (left_dist + right_dist) * 2

    if scale > 0:
        filtered_keypoints[:, :2] /= scale

    return filtered_keypoints


output_filename = f"../../dataset/videos/validation-videos/gmm_classification_output_{buffer_size}_{n_classes}_ok.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_filename, fourcc, 30, (1280, 720))

# Open the video file or webcam
# cap = cv2.VideoCapture("../../dataset/videos/validation-videos/output.mp4")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Error: Unable to access the video source.")
    exit()

print("Processing video... Press 'q' to stop.")


pose_buffer = []

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
            filtered_keypoints = keypoints[i, relevant_indices_normalization, :2]
            normalized_keypoints = normalize_pose_np(filtered_keypoints)

            pose_buffer.append(normalized_keypoints[[relevant_keypoints_normalization.index(kp) for kp in relevant_keypoints]].flatten())
            if len(pose_buffer) > buffer_size:
                pose_buffer.pop(0)

            if len(pose_buffer) == buffer_size:
                # Measure GMM inference time
                start_time = time.time()
                buffer_vector = np.concatenate(pose_buffer)
                buffer_vector_bis = np.copy(buffer_vector)
               
                guard_predicted_class = gmm_guard.predict(buffer_vector.reshape(1, -1))[0]  # Predict class
                guard_class_probabilities = gmm_guard.predict_proba(buffer_vector.reshape(1, -1))[0]  # Get probabilities for each class

                strike_predicted_class = gmm_strike.predict(buffer_vector.reshape(1, -1))[0]  # Predict class
                strike_class_probabilities = gmm_strike.predict_proba(buffer_vector.reshape(1, -1))[0]  # Get probabilities for each class

                predicted_class = 0
                if guard_predicted_class == 1:  #Si strike detecté, alors c'est strike.
                    predicted_class = 1
                elif strike_predicted_class == 0:
                    predicted_class = 0
                else:
                    predicted_class = 2

                class_probabilities = [strike_class_probabilities[0], guard_class_probabilities[1], guard_class_probabilities[0]]

                gmm_inference_time = (time.time() - start_time) * 1000  # ms
                # Define class messages
                class_labels = ["STRIKING !!", "GUARD", "NOT BOXING"]
                class_colors = [(255, 0, 0)] * len(class_labels)  # Default red for all classes
                class_colors[predicted_class] = (0, 255, 0)  # Green for predicted class

                # Overlay all class labels with their probabilities
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.9
                thickness = 2

                text_x, text_y, text_size = 0, 0, 0
                for idx, label in enumerate(class_labels):
                    class_text = f"{label}: {class_probabilities[idx]:.2f}"
                    text_size, _ = cv2.getTextSize(class_text, font, font_scale, thickness)
                    text_x = frame.shape[1] - text_size[0] - 10
                    text_y = 30 + idx * (text_size[1] + 10)
                    cv2.putText(frame, class_text, (text_x, text_y), font, font_scale, class_colors[idx], thickness)

                text_x = frame.shape[1] - text_size[0] - 10
                text_y = 30 + 4 * (text_size[1] + 10)
                class_text = f"Inference: {gmm_inference_time:.2f} ms"
                cv2.putText(frame, class_text, (text_x, text_y), font, font_scale,(255, 0, 0), thickness)
                print(f"GMM Inference Time: {gmm_inference_time:.2f} ms, Predicted Class: {predicted_class}, Probabilities: {class_probabilities}")
            break

    # Display skeleton and frame
    frame = draw_skeleton(frame, keypoints, scores, kpt_thr=0.5)
    cv2.imshow('Pose Estimation and Classification', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    out.write(frame)

out.release()
cap.release()
cv2.destroyAllWindows()
