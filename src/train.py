# train.py
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, metrics, callbacks

# Konfigurasi dasar
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10

train_dir = "data/train"
val_dir = "data/valid"

# Preprocessing
train_gen = ImageDataGenerator(rescale=1./255)
val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

val_data = val_gen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

# Base Model
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Model Functional API
inputs = layers.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs, outputs)

# Compile
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        metrics.Precision(name="precision"),
        metrics.Recall(name="recall")
    ]
)

# Callback: Simpan model terbaik
os.makedirs("models", exist_ok=True)
checkpoint = callbacks.ModelCheckpoint(
    "models/lidah_best_model.h5",
    save_best_only=True,
    monitor="val_accuracy",
    mode="max"
)

# Training
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[checkpoint]
)

# Simpan model terakhir
model.save("models/lidah_model.h5")
print("✅ Training selesai. Model tersimpan di models/lidah_model.h5 dan lidah_best_model.h5")

# Hitung F1-score terakhir dari validation
precision = history.history["val_precision"][-1]
recall = history.history["val_recall"][-1]
f1 = 2 * (precision * recall) / (precision + recall)
print(f"📊 Validation F1-score terakhir: {f1:.4f}")
