import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

_SRC_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_SRC_DIR, "..", "model", "best_model.h5")
IMAGE_SIZE = (224, 224)

CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["space", "del", "nothing"]

print(f"[inference] Loading model from: {os.path.abspath(MODEL_PATH)}")
_model = load_model(MODEL_PATH, compile=False)
print("[inference] Model ready")

@tf.function(reduce_retracing=True)
def _forward(x):
    return _model(x, training=False)


def predict(hand_roi: np.ndarray):
    img = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMAGE_SIZE)
    img = np.expand_dims(img, axis=0).astype(np.float32)
    img = preprocess_input(img)

    predictions = _forward(img).numpy()[0]
    class_idx   = int(np.argmax(predictions))
    return CLASSES[class_idx], float(predictions[class_idx])
