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

# # Keypoints to keep for the GMM
# relevant_keypoints = [
#     'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
#     'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
#     'left_knee', 'right_knee'
# ]

relevant_keypoints = [
                        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
                     ]
    
    
relevant_indices = [coco17_labels.index(kp) for kp in relevant_keypoints]

# Load the GMM model
gmm_model_path = "../../models/gmm/shadow_boxing_gmm_window.pkl"
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
    scale = (left_dist + right_dist) / 2

    if scale > 0:
        filtered_keypoints[:, :2] /= scale

    return filtered_keypoints


trial_no = 5
output_filename = f"../../dataset/videos/validation-videos/gmm_result_output{trial_no}.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_filename, fourcc, 30, (1920, 1080))

# Open the video file or webcam
cap = cv2.VideoCapture(f"../../dataset/videos/validation-videos/videos/output{trial_no}.mp4")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

if not cap.isOpened():
    print("Error: Unable to access the video source.")
    exit()

print("Processing video... Press 'q' to stop.")

# Initialize pose buffer for the sliding window
buffer_size = 30
pose_buffer = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or unable to capture frame.")
        break

    # Perform pose estimation
    keypoints, scores = pose_model(frame)

    # Initialize log-likelihood
    log_likelihood_str = "Log-Likelihood: N/A"
    likelihood_color = (255, 0, 0)  # Blue

    if keypoints is not None and scores is not None:
        for i in range(keypoints.shape[0]):  # Iterate over detected poses
            filtered_keypoints = keypoints[i, relevant_indices, :2]
            normalized_keypoints = normalize_pose_np(filtered_keypoints)

            pose_buffer.append(normalized_keypoints.flatten())
            if len(pose_buffer) > buffer_size:
                pose_buffer.pop(0)

            if len(pose_buffer) == buffer_size:
                # Measure GMM inference time
                start_time = time.time()
                buffer_vector = np.concatenate(pose_buffer)
                log_likelihood = gmm.score(buffer_vector.reshape(1, -1))
                gmm_inference_time = (time.time() - start_time) * 1000  # ms

                log_likelihood_str = f"Log-Likelihood: {log_likelihood:.2f}"
                print(f"GMM Inference Time: {gmm_inference_time:.2f} ms")

                # Decide message based on log-likelihood
                if log_likelihood < 450:
                    message = "Not Boxing"
                    message_color = (0, 0, 255)  # Red
                else:
                    message = "Boxing"
                    message_color = (0, 255, 0)  # Green
            break

    # Overlay log-likelihood
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    thickness = 2

    # Log-likelihood
    text_size, _ = cv2.getTextSize(log_likelihood_str, font, font_scale, thickness)
    text_x = frame.shape[1] - text_size[0] - 10
    text_y = 30
    cv2.putText(frame, log_likelihood_str, (text_x, text_y), font, font_scale, likelihood_color, thickness)

    # Message
    if 'message' in locals():
        message_text_size, _ = cv2.getTextSize(message, font, font_scale, thickness)
        message_x = frame.shape[1] - message_text_size[0] - 10
        message_y = text_y + 30  # Position below the log-likelihood
        cv2.putText(frame, message, (message_x, message_y), font, font_scale, message_color, thickness)

    # Display skeleton and frame
    frame = draw_skeleton(frame, keypoints, scores, kpt_thr=0.8)
    cv2.imshow('Pose Estimation and Log-Likelihood', frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
out.release()
cap.release()
cv2.destroyAllWindows()
