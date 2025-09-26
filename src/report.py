import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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
    class_mode="binary",
    shuffle=False 
)

# Load model
model_path = "models/lidah_model.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model {model_path} tidak ditemukan. Jalankan train.py dulu!")

model = tf.keras.models.load_model(model_path)

# Prediksi
y_true = val_data.classes
y_pred = (model.predict(val_data) > 0.5).astype("int32").flatten()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=val_data.class_indices.keys()))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=val_data.class_indices.keys(),
            yticklabels=val_data.class_indices.keys())
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")

# Simpan gambar
output_path = "confusion_matrix.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"📂 Confusion matrix disimpan ke {output_path}")

# Tampilkan juga
plt.show()
