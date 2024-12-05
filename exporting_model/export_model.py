import tensorflow as tf
import tensorflowjs as tfjs
from constants import MODEL_PATH, MODEL_PATH2, MODEL_FOLDER_PATH

# Cargar Modelo Keras
model = tf.keras.models.load_model(MODEL_PATH2)

# Convertir el modelo a formato TensorFlow.js
tfjs.converters.save_keras_model(model, MODEL_FOLDER_PATH)

print(f"Modelo convertido y guardado en: {MODEL_FOLDER_PATH}")