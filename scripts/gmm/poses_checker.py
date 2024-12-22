import pandas as pd
import numpy as np
import cv2

# Import your custom draw function
from rtmlib import draw_skeleton

############################################
# 1) Load your pose data
############################################


def load_pose_data_of_interest(n_files, movement_name, score_threshold=0.7):
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

            original_len = len(df)
            df = df[df['pose_score'] >= score_threshold]
            removed_count = original_len - len(df)
            print(f"[DEBUG] Removed {removed_count} rows with pose_score < {score_threshold} in file {filename}")
        else:
            print(f"[DEBUG] No 'pose_score' column found in file {filename}")

        all_dfs.append(df)

    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Total number of points in the dataset (pose_score >= {score_threshold}): {len(combined_df)}")
    return combined_df

############################################
# 2) Define helper functions to parse data
#    and draw on a black image
############################################

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

def draw_poses_on_black_image(combined_df):
    """
    Creates a black HD image and draws each pose from combined_df
    on top of it using draw_skeleton.
    """
    # Create a black image in HD resolution (width=1280, height=720)
    width, height = 1280, 720
    

    # Each row in combined_df is one "pose".
    # We'll parse each pose into the format (N, K, 2) or (N, K, 3).
    # Here, N=1 for a single instance, K=17 for coco17.
    for idx, row in combined_df.iterrows():
        black_image = np.zeros((height, width, 3), dtype=np.uint8)
        # Gather keypoints + scores for this row
        keypoints_list = []
        scores_list = []
        for kp_name in COCO_KEYPOINTS:
            x, y, s = get_keypoint_xy_and_score(row, kp_name)
            keypoints_list.append([x, y])
            scores_list.append(s)

        # Convert to numpy arrays
        keypoints_array = np.array(keypoints_list, dtype=np.float32)   # shape: (17,2)
        scores_array    = np.array(scores_list,  dtype=np.float32)     # shape: (17,)

        # Draw the skeleton for this instance onto black_image
        # The function expects shape: (N, K, 2) for keypoints, (N, K) for scores
        black_image = draw_skeleton(
            img=black_image,
            keypoints=keypoints_array[None, :],  # shape becomes (1, 17, 2)
            scores=scores_array[None, :],        # shape becomes (1, 17)
            openpose_skeleton=False,             # Use COCO skeletons
            kpt_thr=0.5,                         # Only draw if score >= 0.5
            radius=3,
            line_width=2
        )

        # Finally, show the image with all poses
        cv2.imshow("All Poses on Black Image", black_image)
        cv2.waitKey(15)
    cv2.destroyAllWindows()

############################################
# 3) Main script
############################################

if __name__ == "__main__":
    # Load your combined DataFrame
    combined_df = load_pose_data_of_interest(21, "shadow")
    draw_poses_on_black_image(combined_df)
