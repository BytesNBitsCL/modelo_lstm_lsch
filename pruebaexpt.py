import tensorflow as tf
import tf2onnx

# Guarda el modelo en formato SavedModel
model = tf.keras.models.load_model(r'C:\Users\alfre\OneDrive\Escritorio\SUF\modelo_lstm_lsch\models\actions_20.keras')
model.save("saved_model")

# Convertir el modelo SavedModel a ONNX
onnx_model, _ = tf2onnx.convert.from_saved_model("saved_model", opset=13)

# Guardar el modelo en formato ONNX
with open("modelo.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
