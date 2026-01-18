import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

GT_DIR = "output/video_faces_ground_truth"
PRED_DIR = "output/video_faces_classified"

CLASSES = ["mask", "sunglass", "mask_sunglasses", "neutral"]

y_true = []
y_pred = []

for cls in CLASSES:
    gt_cls_dir = os.path.join(GT_DIR, cls)

    if not os.path.exists(gt_cls_dir):
        print(f"⚠️ Missing folder: {gt_cls_dir}")
        continue

    for img in os.listdir(gt_cls_dir):
        gt_path = os.path.join(gt_cls_dir, img)

        for pred_cls in CLASSES:
            pred_path = os.path.join(PRED_DIR, pred_cls, img)
            if os.path.exists(pred_path):
                y_true.append(cls)
                y_pred.append(pred_cls)
                break

if len(y_true) == 0:
    raise ValueError("❌ No matched images found. Label ground truth images first.")

cm = confusion_matrix(y_true, y_pred, labels=CLASSES)

plt.figure(figsize=(7,6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASSES,
    yticklabels=CLASSES
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Video Face Classification – Confusion Matrix")
plt.tight_layout()
plt.savefig("output/video_confusion_matrix.png")
plt.show()

print("✅ Confusion matrix saved to output/video_confusion_matrix.png")
