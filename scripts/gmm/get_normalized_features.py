import pandas as pd
import numpy as np
import cv2

# Import your custom draw function
from rtmlib import draw_skeleton

############################################
# 1) Load your pose data
############################################

def load_pose_data_of_interest(n_files, movement_name, score_threshold=0.6):
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

def calculate_angle(v1, v2):
    """Calculate the angle between two vectors."""
    dot_product = np.dot(v1, v2)
    magnitude = np.linalg.norm(v1) * np.linalg.norm(v2)
    if magnitude == 0:
        return 0.0
    angle = np.arccos(np.clip(dot_product / magnitude, -1.0, 1.0))
    return angle / 100  # Normalize angle by dividing by 100  # Normalize angle and ensure it stays between 0 and 1  # Normalize angle in radians

def calculate_keypoint_speeds(df):
    """Calculate normalized speed for each 2D keypoint."""
    speeds = []
    for idx in range(len(df)):
        if idx == 0:
            speeds.append({f"speed_{col}": 0 for col in df.columns if '_x' in col or '_y' in col})
            continue

        row_speed = {}
        for col in df.columns:
            if '_x' in col or '_y' in col:
                keypoint = col.rsplit('_', 1)[0]
                dx = df.at[idx, f'{keypoint}_x'] - df.at[idx - 1, f'{keypoint}_x']
                dy = df.at[idx, f'{keypoint}_y'] - df.at[idx - 1, f'{keypoint}_y']
                norm = np.sqrt(dx**2 + dy**2)
                row_speed[f"speed_{col}"] = norm / 100  # Normalize by dividing by 1000

        speeds.append(row_speed)

    return pd.DataFrame(speeds)

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
    left_elbow = ['left_elbow_x', 'left_elbow_y']
    right_elbow = ['right_elbow_x', 'right_elbow_y']
    left_wrist = ['left_wrist_x', 'left_wrist_y']
    right_wrist = ['right_wrist_x', 'right_wrist_y']

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
        scale = (left_dist + right_dist)*2

        # Avoid division by zero
        if scale > 0:
            for col in translated_pose.index:
                if '_x' in col or '_y' in col:
                    translated_pose[col] /= scale

        # Calculate angles
        left_upper_arm = [
            translated_pose[left_elbow[0]] - translated_pose[left_shoulder[0]],
            translated_pose[left_elbow[1]] - translated_pose[left_shoulder[1]]
        ]
        left_lower_arm = [
            translated_pose[left_wrist[0]] - translated_pose[left_elbow[0]],
            translated_pose[left_wrist[1]] - translated_pose[left_elbow[1]]
        ]
        right_upper_arm = [
            translated_pose[right_elbow[0]] - translated_pose[right_shoulder[0]],
            translated_pose[right_elbow[1]] - translated_pose[right_shoulder[1]]
        ]
        right_lower_arm = [
            translated_pose[right_wrist[0]] - translated_pose[right_elbow[0]],
            translated_pose[right_wrist[1]] - translated_pose[right_elbow[1]]
        ]

        translated_pose['left_arm_angle'] = calculate_angle(left_upper_arm, left_lower_arm)
        translated_pose['right_arm_angle'] = calculate_angle(right_upper_arm, right_lower_arm)

        normalized_data.append(translated_pose)

    return pd.DataFrame(normalized_data)


def save_normalized_data(normalized_df, filepath):
    normalized_df.to_csv(filepath, index=False)
    print(f"Normalized data saved to {filepath}")

def load_normalized_data(filepath):
    return pd.read_csv(filepath)

############################################
# 3) Main script
############################################

if __name__ == "__main__":
    combined_df = load_pose_data_of_interest(83, "shadow")
    normalized_df = normalize_pose(combined_df)
    speeds_df = calculate_keypoint_speeds(combined_df)

    # Merge normalized poses with speeds
    normalized_df = pd.concat([normalized_df.reset_index(drop=True), speeds_df.reset_index(drop=True)], axis=1)

    # Save the normalized data
    save_path = "../../dataset/2D-poses/shadow/shadow_dataset_normalized.csv"
    save_normalized_data(normalized_df, save_path)

    print(f"Normalized data with angles and speeds saved to {save_path}")





