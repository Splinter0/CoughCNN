import os
import json
import librosa
import argparse
import numpy as np
import tensorflow.keras as keras
from processing import load_audio, SAMPLE_SIZE, melspectrogram, SR

class Detector(object):
    def __init__(self, m="model.json", w="model.h5"):
        self.m = m
        self.w = w

        json_file = open(self.m, 'r')
        self.model = json_file.read()
        json_file.close()
        self.model = keras.models.model_from_json(self.model)
        self.model.load_weights(self.w)
#        os.system("clear")

    def detect(self, audio, out, visual):
        self.out = out
        try:
            signal = load_audio(audio)

        except ValueError:
            if visual:
                print("Recording too short!")
            return

        if len(signal) < SAMPLE_SIZE:
            if visual:
                print("Recording too short!")
            return

        current = 0
        end = False
        predictions = []
        pieces = []

        while not end:
            if current+SAMPLE_SIZE > len(signal):
                sample = signal[len(signal)-SAMPLE_SIZE:]
                end = True
            else:
                sample = signal[current:current+SAMPLE_SIZE]
                current += SAMPLE_SIZE

            mel = melspectrogram(sample)
            x = np.array(mel)
            x = x[np.newaxis, ...]
            x = np.expand_dims(x, axis=3)
            pred = np.argmax(self.model.predict(x), axis=1)
            predictions.append(pred)
            pieces.append(sample)

        for i in range(len(predictions)):
            if predictions[i][0] == 0:
                librosa.output.write_wav(self.out + str(i) + os.path.split(audio)[1], pieces[i], sr=SR)

        if visual:
            end = '\033[0m'
            green = '\033[92m'
            length = 50
            each = "|" * (length // len(predictions))
            output = ""
            for p in predictions:
                if p[0] == 0:
                    output += green + each + end
                else:
                    output += each

            print(output)
        else:
            return pieces, predictions


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Classify where the cough is in a file and split it up")
    parser.add_argument("file", metavar="f", type=str, help="Path to the audio file, must be wav")
    args = parser.parse_args()

    try:
        os.system("mkdir detected/")
    except:
        pass

    d = Detector("model.json", "model.h5")
    d.detect(args.file, "detected/", visual=True)
