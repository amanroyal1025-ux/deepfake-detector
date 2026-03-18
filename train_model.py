import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import cv2

# Path to dataset
REAL_PATH = "dataset/real"
FAKE_PATH = "dataset/fake"

IMG_SIZE = 128

data = []
labels = []

def load_images(path, label):
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0
        data.append(img)
        labels.append(label)

# Load dataset
load_images(REAL_PATH, 0)   # Real
load_images(FAKE_PATH, 1)   # Fake

# Convert to numpy
X = np.array(data)
y = np.array(labels)

# Build small CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(X, y, epochs=5, batch_size=16)

# Save model
os.makedirs("model", exist_ok=True)
model.save("model/model.h5")

print("Model saved successfully!")
