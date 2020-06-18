import os
import cv2
import wandb
import datetime
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, LeakyReLU, Activation, Flatten, Dense, Reshape, Input

DATASET = "dataset.npy"

def load_data(f):
    data = np.load(f, allow_pickle=True)
    data = np.array(data)
    train = []
    validation = []

    for sample in data:
        if sample[0] == 0:
            validation.append(sample[1])
        else:
            train.append(sample[1])

    start = len(train)-int(len(train)*0.3)
    validation += train[start:]
    train = train[:start]
    train = np.array(train)
    validation = np.array(validation)
    np.random.shuffle(validation)

    shape = (train.shape[1], train.shape[2], 1)

    train = train.reshape(train.shape[0], shape[0], shape[1], shape[2])
    validation = validation.reshape(validation.shape[0], shape[0], shape[1], shape[2])
    return train, validation, shape

def MSE(x, y):
    err = np.sum((x.astype("float") - y.astype("float")) ** 2)
    err /= float(x * y)
    return err

def visualize_predictions(decoded, gt, samples=10):
	outputs = None
	for i in range(0, samples):
		original = (gt[i] * 255).astype("uint8")
		recon = (decoded[i] * 255).astype("uint8")
		output = np.hstack([original, recon])
		if outputs is None:
			outputs = output
		else:
			outputs = np.vstack([outputs, output])
	return outputs

class AutoEncoder(object):
    def __init__(self, name, config, hyper=False, hyper_project="", extra=None):
        self.name = name
        self.config = config

        if hyper:
            wandb.init(config=config, project=hyper_project)
            wandb.run.save()
            try:
                os.system("mkdir sweep/"+wandb.run.name)
            except:
                pass

    def build(self, input_shape):
        input_layer = Input(shape=input_shape)
        x = input_layer
        x = Conv2D(self.config["conv1"], self.config["kernel1"], strides=2, padding="same")(x)
        x = LeakyReLU(alpha=self.config["alpha"])(x)
        x = BatchNormalization()(x)

        x = Conv2D(self.config["conv2"], self.config["kernel2"], strides=2, padding="same")(x)
        x = LeakyReLU(alpha=self.config["alpha"])(x)
        x = BatchNormalization()(x)

        x = Conv2D(self.config["conv3"], self.config["kernel3"], strides=2, padding="same")(x)
        x = LeakyReLU(alpha=self.config["alpha"])(x)
        x = BatchNormalization()(x)

        magnitude = K.int_shape(x)
        x = Flatten()(x)

        hand = Dense(self.config["dense"])(x)

        self.encoder = Model(input_layer, hand, name="encoder")

        # decoder
        decoder_input = Input(shape=(self.config["dense"],))
        x = Dense(np.prod(magnitude[1:]))(decoder_input)
        x = Reshape((magnitude[1], magnitude[2], magnitude[3]))(x)

        x = Conv2DTranspose(self.config["conv3"], self.config["kernel3"], strides=2, padding="same")(x)
        x = LeakyReLU(alpha=self.config["alpha"])(x)
        x = BatchNormalization()(x)

        x = Conv2DTranspose(self.config["conv2"], self.config["kernel2"], strides=2, padding="same")(x)
        x = LeakyReLU(alpha=self.config["alpha"])(x)
        x = BatchNormalization()(x)


        x = Conv2DTranspose(self.config["conv1"], self.config["kernel1"], strides=2, padding="same")(x)
        x = LeakyReLU(alpha=self.config["alpha"])(x)
        x = BatchNormalization()(x)

        x = Conv2DTranspose(self.config["trans"], self.config["kernel1"], padding="same")(x)
        outputs = Activation("sigmoid")(x)

        self.decoder = Model(decoder_input, outputs, name="decoder")

        self.model = Model(input_layer, self.decoder(self.encoder(input_layer)), name="autoencoder")

    def train(self, x_train, x_test):
        self.optimizer = Adam(lr=self.config["lr"], decay=self.config["lr"]/self.config["epochs"])
        self.model.compile(loss="mse", optimizer=self.optimizer, metrics=['accuracy'])
        self.model.summary()
        self.model.fit(
            x_train, x_train,
            validation_data=(x_test, x_test),
            epochs=self.config["epochs"],
            batch_size=self.config["batch_size"]
        )

    def test(self, x_test):
        decoded = self.model.predict(x_test)
        vis = visualize_predictions(decoded, x_test)
        cv2.imwrite("wow.png", vis)

    def getNormal(self, validation):
        errors = []
        for sample in validation:
            pred = self.model.predict(sample)
            err = MSE(pred, sample)
            errors.append(err)

        error_df = pd.DataFrame({'reconstruction_error':errors})
        error_df.describe()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        _ = ax.hist(error_df.reconstruction_error.values, bins=5)
        fig.show()


if __name__ == '__main__':
    train, validation, shape = load_data(DATASET)
    print(shape)
    #print(x[0], y[0], shape)
    config = dict(
        conv1 = 16,
        conv2 = 32,
        conv3 = shape[0],
        kernel1 = (3,3),
        kernel2 = (3,3),
        kernel3 = (3,3),

        dense = 16,
        trans = shape[-1],

        batch_size = 32,
        epochs = 30,
        alpha = 0.1,
        lr = 1e-4
    )
    m = AutoEncoder("Covid1", config, hyper=False, hyper_project="CovidDetection")
    m.build(shape)
    m.train(train, validation)
    m.test(validation)
    m.getNormal(validation)
