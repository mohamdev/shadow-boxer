import cv2
import time
import csv
from rtmlib import RTMPose, RTMO, draw_skeleton

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

# Initialize the RTMPose model
pose_model = RTMO(
    onnx_model='../../models/pose_estimation/rtmo-l.onnx', #replace by your .onnx rtmo model
    model_input_size=(640, 640),
    backend=backend,
    device=device
)

trial_no = "4"
# Open the video file or webcam
cap = cv2.VideoCapture("../../dataset/videos/shadow/shadow" + trial_no + ".mp4")

if not cap.isOpened():
    print("Error: Unable to access the video source.")
    exit()

print("Processing video... Press 'q' to stop.")

# Open CSV file for writing
csv_file = open("../../dataset/2D-poses/shadow/shadow" + trial_no + ".csv", mode='w', newline='')
csv_writer = csv.writer(csv_file)

# Write the header row to the CSV
header = []
for label in coco17_labels:
    header.extend([f"{label}_x", f"{label}_y"])
header.append("pose_score")
csv_writer.writerow(header)

# Process the video stream
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or unable to capture frame.")
        break

    # Get the resolution of the current frame
    resolution = (frame.shape[1], frame.shape[0])  # (width, height)

    # Measure inference start time
    start_time = time.time()

    # Perform pose estimation
    keypoints, scores = pose_model(frame)

    # Measure inference end time
    end_time = time.time()
    inference_time = (end_time - start_time) * 1000  # Convert to milliseconds

    # Print resolution and inference time
    print(f"Resolution: {resolution[0]}x{resolution[1]} | Inference Time: {inference_time:.2f} ms")

    # Write keypoints and scores to CSV
    if keypoints is not None and scores is not None:
        for i in range(keypoints.shape[0]):  # Iterate over batch
            row = []
            for j in range(keypoints.shape[1]):  # Iterate over keypoints
                row.extend([keypoints[i, j, 0], keypoints[i, j, 1]])
            row.append(scores[i].mean() if len(scores[i]) > 0 else 0.0)  # Add pose score
            csv_writer.writerow(row)

    # Visualize the results
    frame = draw_skeleton(frame, keypoints, scores, kpt_thr=0.5)

    # Display the frame
    cv2.imshow('Pose Estimation', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close CSV file
csv_file.close()

# Release the video source and close windows
cap.release()
cv2.destroyAllWindows()
