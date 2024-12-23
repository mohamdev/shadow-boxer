import pandas as pd
import numpy as np
import cv2

# Import your custom draw function
from rtmlib import draw_skeleton

############################################
# 1) Load your pose data
############################################

def load_pose_data_of_interest(n_files, movement_name, score_threshold=0.70):
    exclude_keywords = ['nose_', 'eye_', 'ear_']
    all_dfs = []

    for i in range(n_files):
        filename = f"../../dataset/2D-poses/{movement_name}/{movement_name}{i}.csv"
        df = pd.read_csv(filename)

        # Keep only columns that do NOT contain nose_, eye_, or ear_
        filtered_cols = [
            col for col in df.columns
            if not any(keyword in col for keyword in exclude_keywords)
        ]

        # Ensure `pose_score` is retained if it exists
        if 'pose_score' in df.columns and 'pose_score' not in filtered_cols:
            filtered_cols.append('pose_score')

        # Filter columns
        df = df[filtered_cols]

        # Convert 'pose_score' to numeric, if it exists
        if 'pose_score' in df.columns:
            df['pose_score'] = pd.to_numeric(df['pose_score'], errors='coerce')
            df.dropna(subset=['pose_score'], inplace=True)
            df = df[df['pose_score'] >= score_threshold]

        all_dfs.append(df)

    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Total number of points in the dataset (pose_score >= {score_threshold}): {len(combined_df)}")
    return combined_df

############################################
# 2) Normalize and Draw Poses
############################################

def normalize_pose(pose_df):
    """
    Normalize each pose to be invariant to scale and translation, 
    using the mean distance between left_shoulder to left_hip and right_shoulder to right_hip.
    """
    print("Normalizing poses...")
    left_hip = ['left_hip_x', 'left_hip_y']
    right_hip = ['right_hip_x', 'right_hip_y']
    left_shoulder = ['left_shoulder_x', 'left_shoulder_y']
    right_shoulder = ['right_shoulder_x', 'right_shoulder_y']

    normalized_data = []

    for _, row in pose_df.iterrows():
        # Calculate the center of the hips for translation
        hip_center_x = (row[left_hip[0]] + row[right_hip[0]]) / 2
        hip_center_y = (row[left_hip[1]] + row[right_hip[1]]) / 2

        # Translate pose so hip center is at (0, 0)
        translated_pose = row.copy()
        for col in row.index:
            if '_x' in col:
                translated_pose[col] -= hip_center_x
            elif '_y' in col:
                translated_pose[col] -= hip_center_y

        # Calculate distances for scaling
        left_dist = np.sqrt(
            (translated_pose[left_shoulder[0]] - translated_pose[left_hip[0]]) ** 2 +
            (translated_pose[left_shoulder[1]] - translated_pose[left_hip[1]]) ** 2
        )
        right_dist = np.sqrt(
            (translated_pose[right_shoulder[0]] - translated_pose[right_hip[0]]) ** 2 +
            (translated_pose[right_shoulder[1]] - translated_pose[right_hip[1]]) ** 2
        )

        # Use the mean of the distances as the scaling factor
        scale = (left_dist + right_dist) / 2

        # Avoid division by zero
        if scale > 0:
            for col in translated_pose.index:
                if '_x' in col or '_y' in col:
                    translated_pose[col] /= scale

        normalized_data.append(translated_pose)

    return pd.DataFrame(normalized_data)


def rescale_pose_for_display(normalized_df, width=1280, height=720):
    """
    Rescale normalized poses to fit within the display image dimensions.
    """
    scaled_data = normalized_df.copy()

    # Rescale to fit image dimensions
    scale_factor = min(width, height) / 6  # Scale to half the smaller dimension
    for col in scaled_data.columns:
        if '_x' in col:
            scaled_data[col] = scaled_data[col] * scale_factor + width // 2
        elif '_y' in col:
            scaled_data[col] = scaled_data[col] * scale_factor + height // 2

    return scaled_data

# COCO-17 keypoints in standard order
COCO_KEYPOINTS = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

def get_keypoint_xy_and_score(row, kp_name):
    """
    Attempts to retrieve (x, y) for a given kp_name from the row.
    If either x or y column is missing, return (0, 0) with score=0.
    """
    x_col = f'{kp_name}_x'
    y_col = f'{kp_name}_y'

    if x_col in row and y_col in row:
        # If you trust the data is valid, set score = 1.0
        x = float(row[x_col])
        y = float(row[y_col])
        # If either x,y is NaN or obviously invalid, you might set score=0 instead
        return (x, y, 1.0)
    else:
        # Keypoint is missing from the DataFrame (excluded by the filter)
        # Return default (0, 0) with score=0 so it won't draw
        return (0.0, 0.0, 0.0)
    
def draw_poses_on_black_image(rescaled_df):
    width, height = 1280, 720

    for _, row in rescaled_df.iterrows():
        black_image = np.zeros((height, width, 3), dtype=np.uint8)
        keypoints_list = []
        scores_list = []

        for kp_name in COCO_KEYPOINTS:
            x, y, s = get_keypoint_xy_and_score(row, kp_name)
            keypoints_list.append([x, y])
            scores_list.append(s)

        keypoints_array = np.array(keypoints_list, dtype=np.float32)
        scores_array = np.array(scores_list, dtype=np.float32)

        black_image = draw_skeleton(
            img=black_image,
            keypoints=keypoints_array[None, :],
            scores=scores_array[None, :],
            openpose_skeleton=False,
            kpt_thr=0.5,
            radius=3,
            line_width=2
        )

        cv2.imshow("Rescaled Poses", black_image)
        cv2.waitKey(1)
    cv2.destroyAllWindows()

############################################
# 3) Save and Load Normalized Data
############################################

def save_normalized_data(normalized_df, filepath):
    normalized_df.to_csv(filepath, index=False)
    print(f"Normalized data saved to {filepath}")

def load_normalized_data(filepath):
    return pd.read_csv(filepath)

############################################
# 4) Main script
############################################

if __name__ == "__main__":
    combined_df = load_pose_data_of_interest(23, "shadow")
    normalized_df = normalize_pose(combined_df)
    rescaled_df = rescale_pose_for_display(normalized_df)

    # Save the normalized data
    save_path = "../../dataset/2D-poses/shadow/shadow_dataset_normalized.csv"
    save_normalized_data(normalized_df, save_path)

    # Draw the poses
    draw_poses_on_black_image(rescaled_df)
