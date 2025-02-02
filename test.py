import pandas as pd

df = pd.read_csv('video_info.csv')
data = df['num_frames']
# find the minimum number of frames
min_frames = min(data)
# find the 10th percentile of the number of frames
percentile_10 = df['num_frames'].quantile(0.20)
# find the 90th percentile of the number of frames
percentile_90 = df['num_frames'].quantile(0.9)
print(min_frames, percentile_10, percentile_90)