import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import librosa.display
import matplotlib.pyplot as plt
from speechpy.processing import cmvn

SR = 44000
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 64
SILENCE = 0.0018
SAMPLE_LENGTH = 32/86 #s
SAMPLE_SIZE = int(np.ceil(SR*SAMPLE_LENGTH))
NOISE_RATIO = 0.25

LABELS = ["cough", "not"]

AUGMENT = "noise/"
noises = []

def envelope(signal, rate, thresh):
    mask = []
    y = pd.Series(signal).apply(np.abs)
    # Create aggregated mean
    y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean()
    for m in y_mean:
        mask.append(m > thresh)

    return mask

def load_audio(path):
    signal, rate = librosa.load(path, sr=SR)
    mask = envelope(signal, rate, SILENCE)
    signal = signal[mask]

    return signal

def melspectrogram(signal):
    signal = librosa.util.normalize(signal)
    spectro = librosa.feature.melspectrogram(
        signal,
        sr=SR,
        n_mels=N_MELS,
        n_fft=N_FFT
    )
    spectro = librosa.power_to_db(spectro)
    spectro = spectro.astype(np.float32)
    return spectro

def load_noises(n=2):
    ns = []
    ids = []
    for _ in range(n):
        while True:
            i = np.random.choice(len(noises))
            if i in ids:
                continue
            ids.append(i)
            noise, _ = librosa.load(noises[i], sr=SR)
            if len(noise) < SAMPLE_SIZE:
                continue
            ns.append(noise)
            break

    return ns

def augment(sample, ns):
    augmented = []
    for noise in ns:
        gap = len(noise)-len(sample)
        point = 0
        if gap > 0:
            point = np.random.randint(low=0, high=len(noise)-len(sample))
        noise = noise[point:point+len(sample)]
        final = []
        for f in range(len(sample)):
            n = noise[f]*NOISE_RATIO
            final.append(sample[f]+n)

        augmented.append(final)

    return augmented

def process(audio, aug=False):
    signal = load_audio(audio)

    if len(signal) < SAMPLE_SIZE:
        return []

    current = 0
    end = False
    features = []

    if aug:
        ns = load_noises()

    while not end:
        if current+SAMPLE_SIZE > len(signal):
            sample = signal[len(signal)-SAMPLE_SIZE:]
            end = True
        else:
            sample = signal[current:current+SAMPLE_SIZE]
            current += SAMPLE_SIZE

        features.append(melspectrogram(sample))

        if aug:
            signals = augment(sample, ns)
            for s in signals:
                features.append(melspectrogram(s))

    return features

def generate_dataset(folder, aug=False):
    data = [] #contains [mel, label]
    for i, label in enumerate(LABELS):
        print("Processing: "+label)
        for audio in tqdm(os.listdir(folder+label)):
            if os.path.splitext(audio)[-1] != ".wav":
                continue

            features = process(folder+label+"/"+audio, aug=aug and i == 0)
            for feat in features:
                data.append([feat, i])

    return data

if __name__ == '__main__':
    for audio in os.listdir(AUGMENT):
        if os.path.splitext(audio)[-1] != ".wav":
            continue

        noises.append(AUGMENT+audio)

    DATA_FOLDER = "dataset/"
    TEST_FOLDER = "test/"

    data = generate_dataset(DATA_FOLDER, True)

    #extra = generate_dataset(EXTRA_FOLDER)
    #data += extra
    np.random.shuffle(data)
    np.save("dataset.npy", data)

    test = generate_dataset(TEST_FOLDER)
    np.save("test.npy", test)
