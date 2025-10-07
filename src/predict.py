import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import sys
import os

# Load model
model = tf.keras.models.load_model("models/lidah_model.h5")

# Path gambar (bisa diubah sesuai kebutuhan)
img_path = sys.argv[1]  # jalankan script dengan path gambar sebagai argumen

# Preprocessing
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # shape (1, 224, 224, 3)
img_array = img_array / 255.0

# Prediksi
prediction = model.predict(img_array)
class_names = ["prediabet", "non_diabet"]

# Output
predicted_class = class_names[int(prediction[0][0] < 0.5)]  # karena sigmoid
confidence = float(prediction[0][0])
confidence = confidence if predicted_class == "non_diabet" else 1 - confidence

print(f"Prediksi: {predicted_class} ({confidence:.2%} yakin)")
