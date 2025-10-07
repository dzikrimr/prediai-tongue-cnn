# evaluate.py
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# Konfigurasi dasar
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
test_dir = "data/test"

# Load test data
test_gen = ImageDataGenerator(rescale=1./255)
test_data = test_gen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# Load model
model_path = "models/lidah_best_model.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model {model_path} tidak ditemukan. Jalankan train.py dulu!")

model = tf.keras.models.load_model(model_path)

# Prediksi
pred_probs = model.predict(test_data)
pred_labels = (pred_probs > 0.5).astype(int).flatten()

# === TEXTUAL REPORT ===
print("\n=== 📊 Classification Report (Test Set) ===")
report = classification_report(
    test_data.classes,
    pred_labels,
    target_names=list(test_data.class_indices.keys()),
    digits=4
)
print(report)

# Confusion Matrix
conf_mat = confusion_matrix(test_data.classes, pred_labels)
print("\n=== 🔢 Confusion Matrix ===")
print(conf_mat)

# === VISUALIZATION ===
classes = list(test_data.class_indices.keys())

# 1️⃣ Confusion Matrix Heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix (Test Set)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("evaluation_confusion_matrix.png")
plt.close()

# 2️⃣ Precision, Recall, F1-score Bar Plot
report_dict = classification_report(
    test_data.classes,
    pred_labels,
    target_names=classes,
    output_dict=True
)

metrics = ["precision", "recall", "f1-score"]
values = [
    [report_dict[cls][metric] for cls in classes]
    for metric in metrics
]

plt.figure(figsize=(7, 4))
x = np.arange(len(classes))
width = 0.25

for i, metric in enumerate(metrics):
    plt.bar(x + i * width, values[i], width, label=metric.capitalize())

plt.xticks(x + width, classes)
plt.ylim(0, 1)
plt.title("Precision, Recall, and F1-score per Class")
plt.legend()
plt.tight_layout()
plt.savefig("evaluation_metrics_bar.png")
plt.close()

# 3️⃣ Akurasi Keseluruhan
overall_acc = report_dict["accuracy"] * 100
print(f"\n✅ Overall Accuracy: {overall_acc:.2f}%")
print("\n📊 Visualisasi evaluasi disimpan sebagai:")
print(" - evaluation_confusion_matrix.png")
print(" - evaluation_metrics_bar.png")
