import os
import numpy as np
import pandas as pd
from mediapipe.python.solutions.holistic import Holistic
from helpers import *
from constants import *
import json

def create_keypoints(word_id, words_path, hdf_path):
    '''
    ### CREAR KEYPOINTS PARA UNA PALABRA
    Recorre la carpeta de frames de la palabra, guarda sus keypoints en `hdf_path`
    y calcula el promedio de los keypoints para esa palabra.
    '''
    data = pd.DataFrame([])
    frames_path = os.path.join(words_path, word_id)
    keypoints_sequences = []  # Lista para almacenar todas las secuencias de keypoints de la palabra
    
    with Holistic() as holistic:
        print(f'Creando keypoints de "{word_id}"...')
        sample_list = os.listdir(frames_path)
        sample_count = len(sample_list)
        
        for n_sample, sample_name in enumerate(sample_list, start=1):
            sample_path = os.path.join(frames_path, sample_name)
            keypoints_sequence = get_keypoints(holistic, sample_path)
            keypoints_sequences.append(keypoints_sequence)  # Añadimos la secuencia a la lista
            data = insert_keypoints_sequence(data, n_sample, keypoints_sequence)
            print(f"{n_sample}/{sample_count}", end="\r")
    
    # Guardamos los keypoints en el archivo HDF5
    data.to_hdf(hdf_path, key="data", mode="w")
    print(f"\nKeypoints creados! ({sample_count} muestras)")
    
    # Ahora calculamos el promedio de los keypoints para esta palabra
    # Aseguramos que todas las secuencias tengan la misma longitud y forma
    normalized_sequences = []
    for seq in keypoints_sequences:
        normalized_seq = normalize_keypoints(seq, target_length=MODEL_FRAMES)
        normalized_sequences.append(normalized_seq)
    
    # Convertimos la lista de secuencias normalizadas en un array de NumPy
    normalized_sequences = np.array(normalized_sequences)
    
    # Calculamos el promedio a lo largo del eje de las muestras (num_samples)
    average_keypoints = np.mean(normalized_sequences, axis=0)
    
    # Convertimos el promedio a lista para poder serializarlo en JSON
    average_keypoints = average_keypoints.tolist()
    
    # Retornamos el promedio de keypoints
    return average_keypoints

def normalize_keypoints(keypoints, target_length=MODEL_FRAMES):
    current_length = len(keypoints)
    if current_length == target_length:
        return keypoints
    
    indices = np.linspace(0, current_length - 1, target_length)
    interpolated_keypoints = []
    for i in indices:
        lower_idx = int(np.floor(i))
        upper_idx = int(np.ceil(i))
        weight = i - lower_idx
        if lower_idx == upper_idx:
            interpolated_keypoints.append(keypoints[lower_idx])
        else:
            interpolated_point = (1 - weight) * np.array(keypoints[lower_idx]) + weight * np.array(keypoints[upper_idx])
            interpolated_keypoints.append(interpolated_point.tolist())
    
    return interpolated_keypoints

if __name__ == "__main__":
    # Crea las carpetas necesarias
    create_folder(KEYPOINTS_PATH)
    
    # Diccionario para almacenar los promedios de todas las palabras
    all_averages = {}
    
    # GENERAR TODAS LAS PALABRAS
    word_ids = [word for word in os.listdir(os.path.join(ROOT_PATH, FRAME_ACTIONS_PATH))]
    
    for word_id in word_ids:
        hdf_path = os.path.join(KEYPOINTS_PATH, f"{word_id}.h5")
        # Obtenemos el promedio de keypoints para la palabra
        average_keypoints = create_keypoints(word_id, FRAME_ACTIONS_PATH, hdf_path)
        all_averages[word_id] = average_keypoints
    
    # Guardamos todos los promedios en un solo archivo JSON
    averages_path = os.path.join(ROOT_PATH, 'all_averages.json')
    with open(averages_path, 'w') as f:
        json.dump(all_averages, f)
    print(f"Promedios de keypoints guardados en {averages_path}")