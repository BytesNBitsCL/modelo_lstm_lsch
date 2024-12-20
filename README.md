# Este es un Fork del repositorio "modelo_lstm_lsp", creado por "ronvidev".
Este es un modelo de una red neuronal que traduce Lengua de Señas Chilena (LSCh) a texto (y voz). Con el objetivo de ser utilizado en una aplicación web. Utilizamos MediaPipe para obtener los puntos de la seña (keypoints) y para el entrenamiento usamos TensorFlow y Keras.

## SCRIPTS PRINCIPALES
- capture_samples.py → captura las muestras y las ubica en la carpeta frame_actions.
- normalize_samples.py → normaliza las muestras para que todas tengan la misma cantidad de frames (importante).
- create_keypoints.py → crea los keypoints que se usarán en el entrenamiento.
- average_keypoints.py → mejora del script "create_keypoints" donde se crea los keypoints que se usarán en el entrenamiento y además crea un promedio de los mismos, que luego se utilizarán en el feedback del usuario.
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
2. Generar los .h5 (keypoints) de cada palabra y crear un promedio de los mismos con ```python average_keypoints.py```
3. Entrenar el modelo con ```python training_model.py```
4. Realizar pruebas con ```python evaluate_model.py```

## Exportar el modelo para usar en la web
Se puede exportar el modelo para ser utilizado en aplicaciones web, para eso ve más información en el README.MD de la carpeta "exporting_model".

## Video de la explicación del código original (creado por ronvidev):
https://youtu.be/3EK0TxfoAMk

## Observaciones
Tener en consideración que utilizamos y es necesaria la versión Python 3.11 para que funcione todo correctamente.

## INSTALAR DEPENDENCIAS
```pip install -r requirements.txt```

## DESINSTALAR DEPENDENCIAS
```
pip freeze > requirements_to_uninstall.txt
pip uninstall -r requirements_to_uninstall.txt -y
Remove-Item requirements_to_uninstall.txt
```
