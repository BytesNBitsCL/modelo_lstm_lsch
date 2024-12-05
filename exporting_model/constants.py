import os

# SETTINGS
MODEL_FRAMES = 30

ROOT_PATH = os.getcwd()
MODEL_FOLDER_PATH = os.path.join(ROOT_PATH, "models")
MODEL_PATH = os.path.join(MODEL_FOLDER_PATH, f"actions_{MODEL_FRAMES}.keras")
MODEL_PATH2 = os.path.join(MODEL_FOLDER_PATH, f"actions_{MODEL_FRAMES}.h5")