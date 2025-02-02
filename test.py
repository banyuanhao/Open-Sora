import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('video_info.csv')
label_encoder = LabelEncoder()
df['text'] = label_encoder.fit_transform(df['text'])

df.to_csv("video_info.csv", index=False)