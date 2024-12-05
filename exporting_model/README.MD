# Bienvenido
Esta carpeta tiene el script para exportar el modelo de forma web utilizando tensorflow, sigue el paso a paso para lograrlo.

## Crear el env
Es necesario, debido a las versiones, utilizar python3.11
```python3.11 -m venv env```

## Activar el env
```source env/bin/activate```

## Instalar Dependencias
```pip install -r requirements.txt```

## Ejecutar Script
Ejecuta el Script que exporta el modelo.
```python export_model.py```


## Desinstalar Dependencias
```
pip freeze > requirements_to_uninstall.txt
pip uninstall -r requirements_to_uninstall.txt -y
Remove-Item requirements_to_uninstall.txt
```