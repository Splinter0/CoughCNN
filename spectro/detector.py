import os
import json
import librosa
import argparse
import numpy as np
import tensorflow.keras as keras
from processing import load_audio, SAMPLE_SIZE, melspectrogram, SR

json_file = open('model.json', 'r')
model = json_file.read()
json_file.close()
model = keras.models.model_from_json(model)
model.load_weights("model.h5")

os.system("clear")

print("Loaded model from disk")

def detect(audio, export=False):
    signal = load_audio(audio)

    if len(signal) < SAMPLE_SIZE:
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
        pred = np.argmax(model.predict(x), axis=1)
        predictions.append(pred)
        pieces.append(sample)

    if export:
        for i in range(len(predictions)):
            if predictions[i][0] == 0:
                librosa.output.write_wav("detected/" + str(i) + ".wav", pieces[i], sr=SR)

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

    detect(args.file, export=True)
