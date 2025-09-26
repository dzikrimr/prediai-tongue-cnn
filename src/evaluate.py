import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Konfigurasi
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
val_dir = "data/valid"

# Preprocessing
val_gen = ImageDataGenerator(rescale=1./255)

val_data = val_gen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

# Load model
model_path = "models/lidah_model.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model {model_path} tidak ditemukan. Jalankan train.py dulu!")

model = tf.keras.models.load_model(model_path)

# Evaluasi
loss, acc = model.evaluate(val_data)
print(f"📊 Validation Accuracy: {acc:.2%}")
print(f"📉 Validation Loss: {loss:.4f}")

