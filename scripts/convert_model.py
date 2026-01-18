import tensorflow as tf

OLD_MODEL = "models/mobilenetv3_mask_classifier.h5"
NEW_MODEL = "models/mobilenetv3_mask_classifier.keras"

model = tf.keras.models.load_model(OLD_MODEL, compile=False)
model.save(NEW_MODEL)

print("✅ Model converted successfully")
