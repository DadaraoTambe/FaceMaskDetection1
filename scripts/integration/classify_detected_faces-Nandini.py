import os
import cv2
import numpy as np
import tensorflow as tf

# ---------------- CONFIG ----------------
MODEL_PATH = "models/mask_classifier.h5"
INPUT_DIR = "output/video_faces"
OUTPUT_DIR = "output/video_faces_classified"

LABELS = ["mask", "mask_sunglasses", "neutral", "sunglasses"]  # ⚠️ MUST MATCH training
IMG_SIZE = 224
CONF_THRESHOLD = 0.5
# ----------------------------------------

# Create output folders
for label in LABELS + ["uncertain"]:
    os.makedirs(os.path.join(OUTPUT_DIR, label), exist_ok=True)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

image_files = [
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith((".jpg", ".png", ".jpeg"))
]

print(f"🔍 Found {len(image_files)} detected face images")

for img_name in image_files:
    img_path = os.path.join(INPUT_DIR, img_name)

    img_original = cv2.imread(img_path)
    if img_original is None:
        continue

    img = cv2.resize(img_original, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img, verbose=0)
    class_id = np.argmax(preds)
    confidence = np.max(preds)

    label = LABELS[class_id] if confidence >= CONF_THRESHOLD else "uncertain"

    save_path = os.path.join(OUTPUT_DIR, label, img_name)
    cv2.imwrite(save_path, img_original)

    print(f"{img_name} → {label} ({confidence:.2f})")

print("✅ All detected faces classified and saved")
