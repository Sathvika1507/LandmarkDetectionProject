import tensorflow as tf
import numpy as np
import cv2

IMG_SIZE = 96
model = tf.keras.models.load_model("landmark_model.keras", compile=False)



img = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
X = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)

pred = model.predict(X)[0] * IMG_SIZE
img_colored = cv2.cvtColor((img * 255).astype('uint8'), cv2.COLOR_GRAY2BGR)

for i in range(0, len(pred), 2):
    x, y = int(pred[i]), int(pred[i+1])
    cv2.circle(img_colored, (x, y), 2, (0, 255, 0), -1)

cv2.imwrite("prediction_output.jpg", img_colored)
print("Saved as prediction_output.jpg")
