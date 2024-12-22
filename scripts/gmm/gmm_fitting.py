import pandas as pd

def load_pose_data_of_interest(n_files, movement_name):
    # Columns we want to exclude:
    exclude_keywords = ['nose_', 'eye_', 'ear_']
    
    all_dfs = []
    for i in range(n_files):
        filename = f"../../dataset/2D-poses/{movement_name}/{movement_name}{i}.csv"
        df = pd.read_csv(filename)
        
        # Keep only the columns that do NOT contain "nose_", "eye_" or "ear_"
        filtered_cols = [
            col for col in df.columns
            if not any(keyword in col for keyword in exclude_keywords)
        ]
        
        df = df[filtered_cols]
        all_dfs.append(df)
    
    # Concatenate all DataFrames
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Print total number of points (rows)
    print("Total number of points in the dataset:", len(combined_df))
    
    # Return the combined DataFrame for further processing if needed
    return combined_df

if __name__ == "__main__":
    combined_df = load_pose_data_of_interest(21, "shadow")
