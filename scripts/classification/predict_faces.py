# predict_faces.py
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ------------------------
# CONFIG: Update these paths

# ------------------------
# Folder where your model is saved
MODEL_PATH = os.path.abspath("models/mask_classifier.h5")



# Folder with images to predict
TEST_IMAGES_DIR = os.path.abspath("data/raw/mask")

# ------------------------
# CHECK MODEL EXISTS
# ------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

print(f"Loading model from: {MODEL_PATH} ...")
model = load_model(MODEL_PATH, compile=False)
print("Model loaded successfully!")

# ------------------------
# FACE DETECTION SETUP
# ------------------------
# Using OpenCV's pre-trained Haar cascade for face detection
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

if face_cascade.empty():
    raise Exception("Failed to load Haar cascade for face detection!")

# ------------------------
# PREDICTION FUNCTION
# ------------------------
def predict_face(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Cannot read image: {image_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        print(f"No faces detected in {image_path}")
        return

    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (224, 224))  # Adjust size according to your model
        face_img = face_img / 255.0  # Normalize
        face_img = np.expand_dims(face_img, axis=0)

        prediction = model.predict(face_img)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction)

        print(f"Image: {os.path.basename(image_path)} | Predicted Class: {predicted_class} | Confidence: {confidence:.2f}")

        # Draw rectangle and label on the face
        label = f"{predicted_class} ({confidence:.2f})"
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show the image with predictions
    cv2.imshow("Prediction", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ------------------------
# RUN PREDICTIONS ON TEST IMAGES
# ------------------------
if not os.path.exists(TEST_IMAGES_DIR):
    raise FileNotFoundError(f"Test images folder not found: {TEST_IMAGES_DIR}")

image_files = [f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

if len(image_files) == 0:
    print(f"No images found in {TEST_IMAGES_DIR}")
else:
    for img_file in image_files:
        predict_face(os.path.join(TEST_IMAGES_DIR, img_file))
