import cv2
import os
import random
import numpy as np

mask_dir = "data/raw/mask"
glasses_dir = "assets/sunglasses"
out_dir = "data/raw/mask_sunglasses"

os.makedirs(out_dir, exist_ok=True)

mask_imgs = os.listdir(mask_dir)
glasses_imgs = os.listdir(glasses_dir)

for img_name in mask_imgs:
    img = cv2.imread(os.path.join(mask_dir, img_name))
    if img is None:
        continue

    h, w = img.shape[:2]
    glasses = cv2.imread(
        os.path.join(glasses_dir, random.choice(glasses_imgs)),
        cv2.IMREAD_UNCHANGED
    )

    glasses = cv2.resize(glasses, (w // 2, h // 4))

    y1, y2 = h // 4, h // 4 + glasses.shape[0]
    x1, x2 = w // 4, w // 4 + glasses.shape[1]

    if glasses.shape[2] == 4:
        alpha = glasses[:, :, 3] / 255.0
        for c in range(3):
            img[y1:y2, x1:x2, c] = (
                alpha * glasses[:, :, c] +
                (1 - alpha) * img[y1:y2, x1:x2, c]
            )

    cv2.imwrite(os.path.join(out_dir, img_name), img)

print("✅ Mask + Sunglasses dataset created")
