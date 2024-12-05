import os

# SETTINGS
MODEL_FRAMES = 30

ROOT_PATH = os.getcwd()
MAIN_MODEL_FOLDER_PATH = os.path.join(ROOT_PATH, "models/")
MODEL_FOLDER_PATH = os.path.join(ROOT_PATH, "models/webModel")
TFLITE_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'liteModel', 'model.tflite')
MODEL_PATH = os.path.join(MAIN_MODEL_FOLDER_PATH, f"actions_{MODEL_FRAMES}.keras")
MODEL_PATH2 = os.path.join(MAIN_MODEL_FOLDER_PATH, f"actions_{MODEL_FRAMES}.h5")