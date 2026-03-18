from flask import Flask, render_template, request
import os
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
model = tf.keras.models.load_model("model/model.h5")

def predict_video(filepath):
    cap = cv2.VideoCapture(filepath)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (128, 128))
        frame = frame / 255.0
        frames.append(frame)

        if len(frames) == 30:  # limit frames
            break

    frames = np.array(frames)
    frames = np.expand_dims(frames, axis=0)

    prediction = model.predict(frames)[0][0]

    if prediction > 0.5:
        return "Deepfake ❌", prediction
    else:
        return "Real ✅", 1 - prediction


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    result, prob = predict_video(filepath)

    return render_template("index.html", result=result, prob=round(prob * 100, 2))


if __name__ == "__main__":
    app.run(debug=True)
