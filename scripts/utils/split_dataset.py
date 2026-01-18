import os
import shutil
import random

RAW_DIR = "data/cropped"
OUT_DIR = "data/splits"

SPLIT = {"train": 0.7, "val": 0.15, "test": 0.15}

os.makedirs(OUT_DIR, exist_ok=True)

for cls in os.listdir(RAW_DIR):
    cls_path = os.path.join(RAW_DIR, cls)

    # ✅ SAFETY CHECK
    if not os.path.isdir(cls_path):
        continue

    images = os.listdir(cls_path)
    random.shuffle(images)

    n = len(images)
    train_end = int(SPLIT["train"] * n)
    val_end = train_end + int(SPLIT["val"] * n)

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split, files in splits.items():
        split_dir = os.path.join(OUT_DIR, split, cls)
        os.makedirs(split_dir, exist_ok=True)

        for f in files:
            shutil.copy(
                os.path.join(cls_path, f),
                os.path.join(split_dir, f)
            )

print("✅ Dataset split completed")
