#!/usr/bin/env python3

import os
import json
import datetime
import numpy as np
from config import *
import tensorflow as tf
import tensorflow.keras as keras
from pydub import AudioSegment
import librosa, librosa.display
from pydub.silence import split_on_silence
from sklearn.model_selection import train_test_split

os.system("clear")

DATA = "dataset.json"

def load_data(path):
    with open(path, "r") as j:
        data = json.load(j)

    return np.array(data["mfcc"]), np.array(data["labels"])

hyper = dict(
    channel_one = 60,
    kernel_one = (3, 3),
    activation = 'relu',
    pool_one = (3, 3),
    strides_one = (2, 2),
    padding = 'same',

    channel_two = 16,
    kernel_two = (2, 2),
    pool_two = (2, 2),
    strides_two = (2, 2),

    channel_three = 28,
    kernel_three = (2, 2),
    pool_three = (2, 2),
    strides_three = (2, 2),

    deep_one = 497,
    drop_one = 0.5998,
    deep_two = 80,
    drop_two = 0.3656
)

import wandb
from wandb.keras import WandbCallback
wandb.init(config=hyper, project="CoughDetection", name="MainModel")

def model(input_shape):
    m = keras.Sequential()

    # 1st conv layer
    m.add(keras.layers.Conv2D(hyper["channel_one"], hyper["kernel_one"], activation=hyper["activation"], input_shape=input_shape))
    m.add(keras.layers.MaxPooling2D(hyper["pool_one"], strides=hyper["strides_one"], padding='same'))
    m.add(keras.layers.BatchNormalization())

    m.add(keras.layers.Conv2D(hyper["channel_two"], hyper["kernel_two"], activation=hyper["activation"]))
    m.add(keras.layers.MaxPooling2D(hyper["pool_two"], strides=hyper["strides_two"], padding='same'))
    m.add(keras.layers.BatchNormalization())

    m.add(keras.layers.Conv2D(hyper["channel_three"], hyper["kernel_three"], activation=hyper["activation"]))
    m.add(keras.layers.MaxPooling2D(hyper["pool_three"], strides=hyper["strides_three"], padding='same'))
    m.add(keras.layers.BatchNormalization())

    m.add(keras.layers.Flatten())
    m.add(keras.layers.Dense(hyper["deep_one"], activation='relu'))
    m.add(keras.layers.Dropout(hyper["drop_one"]))
    m.add(keras.layers.Dense(hyper["deep_two"], activation='relu'))
    m.add(keras.layers.Dropout(hyper["drop_two"]))

    m.add(keras.layers.Dense(2, activation='softmax'))

    return m

def predict(mod, path):
    mfccs = []
    signal, sr = librosa.load(path, sr=SR)
    # Decide the segments based on length
    segments = len(signal) // SEGMENT_LENGTH
    curr = 0  # For segment indexing
    for segment in range(segments):
        # Extract mfcc data
        mfcc = librosa.feature.mfcc(signal[curr:curr + SEGMENT_LENGTH], sr=SR, n_mfcc=N_MFCC, n_fft=N_FURIER,
                                    hop_length=HOP_LENGTH).T
        if len(mfcc) == EXPECTED_MFCC:
            mfccs.append(mfcc.tolist())

        curr += SEGMENT_LENGTH

    predictions = []
    for x in mfccs:
        x = np.array(x)
        x = x[np.newaxis, ...]
        x = np.expand_dims(x, axis=3)
        pred = np.argmax(mod.predict(x), axis=1)
        predictions.append(pred)

    return predictions

# Pretty Print Prediction... yes
def ppp(prediction):
    end = '\033[0m'
    green = '\033[92m'
    length = 50
    each = "|"*(length//len(prediction))
    output = ""
    for p in prediction:
        if p[0] == 0:
            output += green+each+end
        else:
            output += each

    print(output)

if __name__ == "__main__":
    TRAIN = True

    if TRAIN:
        x, y = load_data(DATA)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3)

        x_train = x_train[..., np.newaxis]
        x_test = x_test[..., np.newaxis]
        x_val = x_val[..., np.newaxis]

        m = model((x_train.shape[1], x_train.shape[2], 1))
        opt = keras.optimizers.Adam(learning_rate=0.000001)

        m.compile(optimizer=opt,
                  loss="sparse_categorical_crossentropy",
                  metrics=['accuracy'])

        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        m.summary()
        m.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=32, epochs=120, callbacks=[WandbCallback()])

        test_err, test_acc = m.evaluate(x_test, y_test, verbose=1)
        print("Accuracy on testing data", test_acc)

        model_json = m.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        print("Saved model to disk")
        m.save_weights("model.h5")

    else:
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = keras.models.model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("model.h5")
        print("Loaded model from disk")

        t = predict(loaded_model, "test2.wav")
        ppp(t)
        t = predict(loaded_model, "test.wav")
        ppp(t)

