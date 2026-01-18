import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from classification.model_utils import build_model

# # ✅ FIXED IMPORT
from model_utils import build_model


model = build_model(num_classes=4)
model.load_weights("models/mobilenetv3_mask_classifier.h5")

test_dir = "data/splits/test"

datagen = ImageDataGenerator(rescale=1.0 / 255)

test_gen = datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

preds = model.predict(test_gen)
y_pred = np.argmax(preds, axis=1)
y_true = test_gen.classes

cm = confusion_matrix(y_true, y_pred)

labels_present = np.unique(y_true)
class_names = list(test_gen.class_indices.keys())
filtered_labels = [class_names[i] for i in labels_present]

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=filtered_labels
)

disp.plot(cmap="Blues", xticks_rotation=45)

# ✅ FIXED SAVE PATH
plt.savefig("output/classification_results/confusion_matrix.png")
plt.show()

print("✅ Confusion matrix saved successfully")
