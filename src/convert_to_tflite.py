"""
Run once:  python src/convert_to_tflite.py
Produces:  model/best_model.tflite  (~5x smaller, 3-5x faster on CPU)
"""
import os
import tensorflow as tf

_SRC = os.path.dirname(os.path.abspath(__file__))
H5_PATH     = os.path.join(_SRC, "..", "model", "best_model.h5")
TFLITE_PATH = os.path.join(_SRC, "..", "model", "best_model.tflite")

print("Loading model…")
model = tf.keras.models.load_model(H5_PATH, compile=False)

print("Converting to TFLite with dynamic-range quantization…")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]   # quantise weights → smaller & faster
tflite_model = converter.convert()

with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

h5_mb     = os.path.getsize(H5_PATH)     / 1024 / 1024
tflite_mb = os.path.getsize(TFLITE_PATH) / 1024 / 1024
print(f"Done!")
print(f"  Original : {h5_mb:.1f} MB")
print(f"  TFLite   : {tflite_mb:.1f} MB  ({h5_mb/tflite_mb:.1f}x smaller)")
print(f"  Saved to : {os.path.abspath(TFLITE_PATH)}")
