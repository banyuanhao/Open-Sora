import pandas as pd
import os
from pathlib import Path
import cv2

# Base path to videos
base_path = Path('/nfs/data/banyuanhao/video/UCF101')
save_path = Path('/data/xuchenheng/UCF101')

# Initialize the CSV file
csv_file = 'video_info.csv'
columns = ['text', 'class_name', 'video_name', 'frame_index', 'width', 'height', 'aspect_ratio', 'resolution', 'fps']

# Create the CSV file with headers if it doesn't exist
if not os.path.exists(csv_file):
    pd.DataFrame(columns=columns).to_csv(csv_file, index=False)

# Iterate through classes and videos
class_names = os.listdir(base_path)

for class_name in class_names:
    if class_name.startswith('.'):
        continue
    path_class = base_path / class_name
    video_names = os.listdir(path_class)
    for video_name in video_names:
        path_video = path_class / video_name
        if video_name.startswith('.'):
            continue
        
        try:
            # Open video file
            cap = cv2.VideoCapture(str(path_video))
            if not cap.isOpened():
                print(f"Error: Could not open the video file {path_video}")
                continue
            
            # Retrieve basic properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            resolution = (width, height)
            aspect_ratio = width / height if height != 0 else None
            
            # Count frames without storing them in memory
            frame_count = 0
            while True:
                ret, _ = cap.read()
                if not ret:
                    break
                frame_count += 1
            
            # Release video resources
            cap.release()

            # Prepare data for the CSV
            data = {
                'path': save_path / class_name / video_name,
                'text': class_name,  
                'num_frames': [frame_count],
                'fps': [fps],
                'height': [height],
                'width': [width],
                'aspect_ratio': [aspect_ratio],
                'resolution': [resolution],
                'text_len': [len(class_name)],
            }
            
            # Append to CSV
            df = pd.DataFrame(data)
            df.to_csv(csv_file, mode='a', header=False, index=False)
            print(f"Saved {class_name}/{video_name} to CSV")
        
        except Exception as e:
            print(f"Error processing {class_name}/{video_name}: {e}")