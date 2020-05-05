import os
import math
import json
import librosa, librosa.display

DATA_FOLDER = "edited_wavs/"
OUTPUT = "dataset.json"
LABELS = ["wet", "dry"]
SR = 16000
N_MFCC = 15
N_FURIER = 2048
HOP_LENGTH = 512
SEGMENT_LENGTH = SR//2
EXPECTED_MFCC = math.ceil(SEGMENT_LENGTH/HOP_LENGTH)

data = {
    "mfcc":[],
    "labels":[]
}

for sample in os.listdir(DATA_FOLDER):
    if os.path.splitext(sample)[-1] != ".wav":
        continue

    label = 0
    name = sample.lower()
    for i in range(len(LABELS)):
        if LABELS[i] in name:
            label = i

    signal, sr = librosa.load(DATA_FOLDER+sample, sr=SR)
    segments = len(signal)//SEGMENT_LENGTH
    print(name, segments)
    curr = 0
    for segment in range(segments):
        mfcc = librosa.feature.mfcc(signal[curr:curr+SEGMENT_LENGTH], sr=SR, n_mfcc=N_MFCC, n_fft=N_FURIER, hop_length=HOP_LENGTH).T
        if len(mfcc) == EXPECTED_MFCC:
            data["mfcc"].append(mfcc.tolist())
            data["labels"].append(label)

        curr += SEGMENT_LENGTH

print(len(data["labels"]))
with open(OUTPUT, "w") as j:
    json.dump(data, j, indent=4)