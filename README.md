Este es un modelo de una red neuronal que traduce Lengua de Señas Peruana (LSP) a texto (y voz). Utilicé MediaPipe para obtener los puntos de la seña y para el entrenamiento usé TensorFlow y Keras.

## SCRIPTS PRINCIPALES
- capture_samples.py → captura las muestras y las ubica en la carpeta frame_actions.
- normalize_samples.py → normaliza las muestras para que todas tengan la misma cantidad de frames (importante).
- create_keypoints.py → crea los keypoints que se usarán en el entrenamiento.
- training_model.py → entrena la red neuronal.
- evaluate_model.py → donde se realiza la prueba de la red neuronal.
- main.py → donde se utiliza una GUI para usar el traductor.

## SCRIPTS SECUNDARIOS
- model.py → aquí se ajusta el modelo de la red neuronal.
- constants.py → ajustes de la red neuronal.
- helpers.py → funciones que se utilizan en los scripts principales.

## Pasos para probar la red neuronal
1. Capturar las muestras con ```python capture_samples.py```
2. Normalizar las muestras con ```python normalize_samples.py```
2. Generar los .h5 (keypoints) de cada palabra con ```python create_keypoints.py```
3. Entrenar el modelo con ```python training_model.py```
4. Realizar pruebas con ```python evaluate_model.py```

## Video de la explicación del código:
https://youtu.be/3EK0TxfoAMk
Nota: Pronto subiré otro video explicando las mejoras.

## INSTALAR DEPENDENCIAS
```pip install -r requirements.txt```

## DESINSTALAR DEPENDENCIAS
```
pip freeze > requirements_to_uninstall.txt
pip uninstall -r requirements_to_uninstall.txt -y
Remove-Item requirements_to_uninstall.txt
```