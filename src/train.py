import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

# Konfigurasi dasar
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10

train_dir = "data/train"
val_dir = "data/valid"

# 1. Preprocessing: Rescaling
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

# 2. Base Model: MobileNetV2
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze weights
base_model.trainable = False

# 3. Bangun model pakai Functional API biar tidak ada multi-input ghost
inputs = layers.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)  # pastikan hanya 1 output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs, outputs)

# 4. Compile
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# 5. Training
model.fit(train_data, epochs=EPOCHS, validation_data=val_data)

# 6. Save
os.makedirs("models", exist_ok=True)
model.save("models/lidah_model.h5")
print("✅ Model trained & saved at models/lidah_model.h5")
