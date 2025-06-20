import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import os

IMG_SIZE = 96
df = pd.read_csv('landmarks_dataset/labels.csv')
X, y = [], []

for idx, row in df.iterrows():
    img_path = os.path.join('landmarks_dataset/images', row['filename'])
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
    X.append(img)
    y.append(row[1:].values.astype('float32') / IMG_SIZE)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(500, activation='relu'),
    tf.keras.layers.Dense(y.shape[1])
])

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=5, batch_size=1)
model.save("landmark_model.keras")



