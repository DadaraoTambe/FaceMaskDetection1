import os
import cv2
from mtcnn import MTCNN

detector = MTCNN()

INPUT_DIR = "data/raw"
CROP_DIR = "data/cropped"

# ✅ FIXED: align with final structure
BOX_DIR = "output/detection_results/images_with_boxes"

categories = ["neutral", "sunglasses", "mask", "mask_sunglasses"]

MIN_SIZE = 50  # skip tiny images

for category in categories:
    in_path = os.path.join(INPUT_DIR, category)
    crop_path = os.path.join(CROP_DIR, category)
    box_path = os.path.join(BOX_DIR, category)

    os.makedirs(crop_path, exist_ok=True)
    os.makedirs(box_path, exist_ok=True)

    for img_name in os.listdir(in_path):
        img_path = os.path.join(in_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        h, w = img.shape[:2]
        if h < MIN_SIZE or w < MIN_SIZE:
            continue  # skip very small images

        try:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(rgb)
        except Exception:
            print(f"⚠️ Skipping {img_name} due to detector error")
            continue

        if len(faces) == 0:
            continue  # no faces found

        for i, face in enumerate(faces):
            x, y, bw, bh = face["box"]
            x, y = max(0, x), max(0, y)

            cropped = img[y:y + bh, x:x + bw]

            if cropped.size == 0:
                continue

            crop_name = f"{img_name.split('.')[0]}_{i}.jpg"
            cv2.imwrite(os.path.join(crop_path, crop_name), cropped)

            # draw bounding box
            cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

        cv2.imwrite(os.path.join(box_path, img_name), img)

print("✅ Face detection & cropping completed safely")
