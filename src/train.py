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

# 2. Model: MobileNetV2 + Head
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Tidak fine-tune dulu

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')  # Binary output
])

# 3. Compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 4. Training
model.fit(train_data, epochs=EPOCHS, validation_data=val_data)

# 5. Save
if not os.path.exists("models"):
    os.makedirs("models")
model.save("models/lidah_model.h5")
print("✅ Model trained & saved at models/lidah_model.h5")
