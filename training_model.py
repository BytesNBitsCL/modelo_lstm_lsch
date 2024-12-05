import numpy as np
from model import get_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from helpers import get_word_ids, get_sequences_and_labels
from constants import *
import matplotlib.pyplot as plt

def training_model(model_path, model_path2, epochs=213): # 500
    word_ids = get_word_ids(WORDS_JSON_PATH) # ['word1', 'word2', 'word3]
    
    sequences, labels = get_sequences_and_labels(word_ids)
    
    sequences = pad_sequences(sequences, maxlen=int(MODEL_FRAMES), padding='pre', truncating='post', dtype='float16')
    
    X = np.array(sequences)
    y = to_categorical(labels).astype(int) 
    
    early_stopping = EarlyStopping(monitor='accuracy', patience=30, restore_best_weights=True)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)
    
    model = get_model(int(MODEL_FRAMES), len(word_ids))
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=32) # callbacks=[early_stopping]
    
    # Precisión de entrenamiento y validación
    plt.figure(figsize=(12, 6))

    # Gráfico de precisión
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Precisión de Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Precisión de Validación')
    plt.title('Precisión en cada Época')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.legend()

    # Gráfico de pérdida
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Pérdida de Entrenamiento')
    plt.plot(history.history['val_loss'], label='Pérdida de Validación')
    plt.title('Pérdida en cada Época')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()

    plt.tight_layout()
    plt.show()

    model.summary()
    model.save(model_path2)
    model.save(model_path)

if __name__ == "__main__":
    training_model(MODEL_PATH, MODEL_PATH2)
    