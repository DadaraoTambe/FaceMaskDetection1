import cv2
import tensorflow as tf
import numpy as np
from collections import Counter
from tensorflow.keras.models import load_model

MODEL_PATH = "models/mobilenetv3_mask_classifier.h5"
VIDEO_PATH = "sample.mp4"

LABELS = ["neutral", "mask", "sunglasses", "mask_sunglasses"]

model = load_model(MODEL_PATH, compile=False, safe_mode=False)

cap = cv2.VideoCapture(VIDEO_PATH)
predictions = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (224, 224))
    frame = frame / 2
