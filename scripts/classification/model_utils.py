import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Hide TF info messages

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import numpy as np


def build_model(num_classes=4):
    """
    Builds a MobileNetV3Small-based model for classification.

    Args:
        num_classes (int): Number of output classes.

    Returns:
        tf.keras.Model: Compiled Keras model.
    """
    # Load pre-trained MobileNetV3Small without top
    base_model = MobileNetV3Small(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False  # Freeze the base

    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    return model


if __name__ == "__main__":
    # Build model
    model = build_model(num_classes=4)

    # Print model architecture
    model.summary()

    # Test model with dummy input
    dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
    dummy_output = model(dummy_input)
    print("\nDummy output shape:", dummy_output.shape)
    print("Dummy output (probabilities):", dummy_output.numpy())
