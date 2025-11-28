import tensorflow as tf
from tensorflow import keras
model = keras.models.load_model('face_classifier.keras')

print(f"Model input shape: {model.input_shape}")
print(f"Model output shape: {model.output_shape}")
print("\nModel architecture:")
model.summary()

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('face_classifier.tflite', 'wb') as f:
    f.write(tflite_model)
print("TensorFlow Lite model created")