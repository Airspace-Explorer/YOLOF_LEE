import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/A/Desktop/json/CVAT_result', header=None)
df.columns=['frame', 'id', 'xtl', 'ytl', 'width', 'height', 'confidence', 'class', "visibility"]
df['confidence'] = 0.8
df['class'] = 0
df['frame_start'] = df['frame_start']

import numpy as np
df['visibility'] = df['visibility'].astype(int)
df['xtl'] = df['xtl'].apply(lambda x: np.round(x,2))
df['ytl'] = df['ytl'].apply(lambda x: np.round(x,2))
df['width'] = df['width'].apply(lambda x: np.round(x, 2))
df['height'] = df['height'].apply(lambda x: np.round(x, 2))

df.to_csv('result.txt', header=None, index=None)
