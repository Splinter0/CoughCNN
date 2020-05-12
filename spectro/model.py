import os
import wandb
import datetime
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from wandb.keras import WandbCallback
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout


DATASET = "dataset.npy"
global model

def load_data(f, s=False):
    data = np.load(f, allow_pickle=True)

    x = np.array([x for x in data[0]])
    y = np.array([to_categorical(y, num_classes=2) for y in data[1]])

    if s:
        return x, y

    shape = (x.shape[1], x.shape[2], x.shape[3])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25)

    return x_train, y_train, x_test, y_test, x_val, y_val, shape

class ExtraCallBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        acc = model.test(model.x_extra, model.y_extra, extra=True)
        try:
            wandb.log({'real_acc': acc})
        except:
            pass
        #print("Evaluating over our own dataset, accuracy: "+str(acc))

class Model(object):
    def __init__(self, name, config, hyper=False, hyper_project="", extra=None):
        self.name = name
        self.config = config
        self.x_extra = extra[0]
        self.y_extra = extra[1]

        if hyper:
            wandb.init(config=config, project=hyper_project)
            self.callback = WandbCallback(data_type="image", validation_data=extra)
        else:
            log_dir = "logs/fit/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        self.extra_callback = ExtraCallBack()

    def build(self, input_shape):
        self.model = keras.Sequential([
            Conv2D(self.config["conv1"], self.config["kernel1"], activation='relu', input_shape=input_shape),
            MaxPooling2D(self.config["pool1"], padding='same'),
            BatchNormalization(),

            Conv2D(self.config["conv2"], self.config["kernel2"], activation='relu'),
            MaxPooling2D(self.config["pool2"], padding='same'),
            BatchNormalization(),

            Flatten(),
            Dense(self.config["deep1"], activation='relu'),
            Dropout(self.config["drop1"]),
            Dense(self.config["deep2"], activation='relu'),
            Dropout(self.config["drop2"]),

            Dense(2, activation='softmax')
        ])

    def train(self, x_train, y_train, validation):
        self.optimizer = keras.optimizers.Adam(learning_rate=self.config["lr"])
        self.model.compile(
            optimizer=self.optimizer,
            loss="categorical_crossentropy",
            metrics=['accuracy']
        )

        self.model.summary()

        self.model.fit(
            x_train,
            y_train,
            validation_data=validation,
            batch_size=self.config["batch_size"],
            epochs=self.config["epochs"],
            callbacks=[self.callback, self.extra_callback]
        )

    def test(self, x_test, y_test, extra=True):
        test_err, test_acc = self.model.evaluate(x_test, y_test, verbose=1)
        if extra:
            return test_acc
        else:
            print("Accuracy on testing data: "+str(test_acc))

    def save(self):
        with open("model.json", "w") as json_file:
            json_file.write(self.model.to_json())

        self.model.save_weights("model.h5")
        print("Saved model '"+self.name+"' to disk")


if __name__ == '__main__':
    should_train = True

    if should_train:
        x_train, y_train, x_test, y_test, x_val, y_val, shape = load_data(DATASET)
        x_extra, y_extra = load_data("test/test.npy", s=True)

        config = dict(
            conv1 = 32,
            kernel1 = (5,5),
            pool1 = (4,4),

            conv2 = 128,
            kernel2 = (5,5),
            pool2 = (2,2),

            deep1 = 128,
            drop1 = 0.6,

            deep2 = 64,
            drop2 = 0.6,

            batch_size = 32,
            epochs = 32,
            lr = 0.00001
        )

        model = Model("Spectro1", config, hyper=True, hyper_project="MelSpect-CoughDetect", extra=(x_extra, y_extra))
        model.build(shape)
        model.train(x_train, y_train, (x_val, y_val))
        model.save()

    else:
        pass
