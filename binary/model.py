import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

DATA = "dataset.json"

def load_data(path):
    with open(path, "r") as j:
        data = json.load(j)

    return np.array(data["mfcc"]), np.array(data["labels"])

def model(input_shape):
    m = keras.Sequential()

    # 1st conv layer
    m.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    m.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    m.add(keras.layers.BatchNormalization())

    # 2nd conv layer
    m.add(keras.layers.Conv2D(32, (2, 2), activation='relu'))
    m.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    m.add(keras.layers.BatchNormalization())

    # 3rd conv layer
    m.add(keras.layers.Conv2D(32, (1, 1), activation='relu'))
    m.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    m.add(keras.layers.BatchNormalization())


    m.add(keras.layers.Flatten())
    m.add(keras.layers.Dense(64, activation='relu'))
    m.add(keras.layers.Dropout(0.3))

    m.add(keras.layers.Dense(2, activation='softmax'))

    return m

if __name__ == "__main__":
    x, y = load_data(DATA)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    print(len(x_train),len(y_train))
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]
    x_val = x_val[..., np.newaxis]

    print(x_train.shape[1], x_train.shape[2], 1)
    m = model((x_train.shape[1], x_train.shape[2], 1))
    opt = keras.optimizers.Adam(learning_rate=0.0001)

    m.compile(optimizer=opt,
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])

    m.summary()
    m.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=32, epochs=30)

    test_err, test_acc = m.evaluate(x_test, y_test, verbose=1)
    print("Accuracy on testing data", test_acc)


