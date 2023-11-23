import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/A/Desktop/json/CVAT_result', header=None)
df.columns = ['frame', 'id', 'xtl', 'ytl', 'width', 'height', 'confidence', 'class', 'visibility']
df['confidence'] = 0.8
df['class'] = 0

import numpy as np
df['id'] = -1
df['xtl'] = df['xtl'].apply(lambda x: np.round(x, 2))
df['ytl'] = df['ytl'].apply(lambda x: np.round(x, 2))
df['width'] = df['width'].apply(lambda x: np.round(x, 2))
df['height'] = df['height'].apply(lambda x: np.round(x, 2))

# Delete 'class' and 'visibility' columns
df = df.drop(columns=['class', 'visibility'], errors='ignore')

df.to_csv('gt.txt', header=None, index=None)
