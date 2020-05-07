import os
import json
import librosa
import argparse
import numpy as np
from config import *
import tensorflow.keras as keras

parser = argparse.ArgumentParser(description="Classify where the cough is in a file and split it up")
parser.add_argument("file", metavar="f", type=str, help="Path to the audio file, must be wav")
args = parser.parse_args()


json_file = open('model.json', 'r')
model = json_file.read()
json_file.close()
model = keras.models.model_from_json(model)
model.load_weights("model.h5")

os.system("clear")

print("Loaded model from disk")

try:
    os.system("mkdir detected/")
except:
    pass

mfccs = []
signal, sr = librosa.load(args.file, sr=SR)
# Decide the segments based on length
segments = len(signal) // SEGMENT_LENGTH
pieces = []
curr = 0  # For segment indexing
for segment in range(segments):
    p = signal[curr:curr + SEGMENT_LENGTH]
    # Extract mfcc data
    mfcc = librosa.feature.mfcc(p, sr=SR, n_mfcc=N_MFCC, n_fft=N_FURIER,
                                hop_length=HOP_LENGTH).T
    if len(mfcc) == EXPECTED_MFCC:
        mfccs.append(mfcc.tolist())
        pieces.append(p)

    curr += SEGMENT_LENGTH

predictions = []
for x in mfccs:
    x = np.array(x)
    x = x[np.newaxis, ...]
    x = np.expand_dims(x, axis=3)
    pred = np.argmax(model.predict(x), axis=1)
    predictions.append(pred)

for i in range(len(predictions)):
    if predictions[i][0] == 0:
        librosa.output.write_wav("detected/"+str(i)+".wav", pieces[i], sr=SR)

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