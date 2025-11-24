import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pathlib

# Base paths
base_dir = pathlib.Path(__file__).resolve().parent
train_dir = base_dir / "data" / "train"
val_dir = base_dir / "data" / "val"

img_size = (128, 128)
batch_size = 32

# Load train / val datasets
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    labels="inferred",          # folder names = class names
    label_mode="binary",        # face / non_face
    batch_size=batch_size,
    image_size=img_size,
    shuffle=True,
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    labels="inferred",
    label_mode="binary",
    batch_size=batch_size,
    image_size=img_size,
    shuffle=False,
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = layers.Rescaling(1.0 / 255)

model = keras.Sequential([
    normalization_layer,
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=img_size + (3,)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid"),   # probability of "face"
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()

epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
)

# Create save directory
save_root = base_dir / "saved_models"
save_root.mkdir(parents=True, exist_ok=True)

# 1) Save as Keras format (.keras) for Python usage
keras_path = save_root / "face_classifier.keras"
model.save(keras_path)
print(f"Keras model saved to: {keras_path}")

# 2) Export as SavedModel directory for C++ / TF serving usage
savedmodel_dir = save_root / "face_classifier_savedmodel"
model.export(savedmodel_dir)
print(f"SavedModel exported to: {savedmodel_dir}")
