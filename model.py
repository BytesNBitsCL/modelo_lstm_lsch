from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from keras.regularizers import l2
from constants import LENGTH_KEYPOINTS

def get_model(max_length_frames, output_length: int):
    model = Sequential()
    
    model.add(Input(batch_shape=(1,max_length_frames, LENGTH_KEYPOINTS)))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(256, return_sequences=False))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(output_length, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model