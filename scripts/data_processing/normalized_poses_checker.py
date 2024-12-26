import pandas as pd
import numpy as np
import cv2
# Import your custom draw function
from rtmlib import draw_skeleton

def rescale_pose_for_display(normalized_df, width=1280, height=720, rescale_factor=10):
    """
    Rescale normalized poses to fit within the display image dimensions.
    """
    scaled_data = normalized_df.copy()

    # Rescale to fit image dimensions
    scale_factor = (min(width, height) / rescale_factor)*4  # Scale to half the smaller dimension
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
        cv2.waitKey(15)
    cv2.destroyAllWindows()


def load_normalized_data(filepath):
    return pd.read_csv(filepath)

############################################
# 4) Main script
############################################

if __name__ == "__main__":

    # Save the normalized data
    # save_path = "../../dataset/2D-poses/shadow/shadow_dataset_poses_normalized.csv"
    save_path = "../../clusters/cluster_-1.csv"

    normalized_df = load_normalized_data(save_path)
    rescaled_df = rescale_pose_for_display(normalized_df)

    # Draw the poses
    draw_poses_on_black_image(rescaled_df)
