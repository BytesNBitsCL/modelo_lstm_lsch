import os
import tensorflow as tf
import tensorflowjs as tfjs
from constants import MODEL_PATH, MODEL_PATH2, MODEL_FOLDER_PATH, TFLITE_MODEL_PATH

# Asegurarse de que el directorio para el modelo TFLite exista
os.makedirs(os.path.dirname(TFLITE_MODEL_PATH), exist_ok=True)

# Cargar Modelo Keras
model = tf.keras.models.load_model(MODEL_PATH2)

# Convertir el modelo a formato TensorFlow.js
tfjs.converters.save_keras_model(model, MODEL_FOLDER_PATH)
print(f"Modelo convertido y guardado en formato TensorFlow.js en: {MODEL_FOLDER_PATH}")

# Convertir el modelo a TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Guardar el modelo TensorFlow Lite
with open(TFLITE_MODEL_PATH, 'wb') as f:
    f.write(tflite_model)

print(f"Modelo convertido y guardado en formato TensorFlow Lite en: {TFLITE_MODEL_PATH}")

# Opcional: Crear una versión cuantizada del modelo TFLite para reducir el tamaño
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quantized_model = converter.convert()

# Guardar el modelo TensorFlow Lite cuantizado
quantized_tflite_model_path = TFLITE_MODEL_PATH.replace('.tflite', '_quantized.tflite')
with open(quantized_tflite_model_path, 'wb') as f:
    f.write(tflite_quantized_model)

print(f"Modelo cuantizado convertido y guardado en formato TensorFlow Lite en: {quantized_tflite_model_path}")

# Imprimir tamaños de los modelos para comparación
tfjs_size = sum(os.path.getsize(os.path.join(MODEL_FOLDER_PATH, f)) for f in os.listdir(MODEL_FOLDER_PATH) if os.path.isfile(os.path.join(MODEL_FOLDER_PATH, f)))
tflite_size = os.path.getsize(TFLITE_MODEL_PATH)
tflite_quantized_size = os.path.getsize(quantized_tflite_model_path)

print(f"\nTamaños de los modelos:")
print(f"TensorFlow.js: {tfjs_size / 1024:.2f} KB")
print(f"TensorFlow Lite: {tflite_size / 1024:.2f} KB")
print(f"TensorFlow Lite (cuantizado): {tflite_quantized_size / 1024:.2f} KB")