import os
import cv2
import librosa
import skimage.io
import numpy as np
import pandas as pd
from tqdm import tqdm
import librosa.display
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

import imageio.core.util

def ignore_warnings(*args, **kwargs):
    pass

imageio.core.util._precision_warn = ignore_warnings

DATA_FOLDER = "test/"
OUT = "test/samples/"
DATASET = "test/test.npy"
LABELS = ["cough", "not"]

sample_rate = 44000
duration = 0.5 #seconds
hop_length = 347*duration
n_mels = 128
fmin = 20
fmax = sample_rate//2
n_fft = n_mels*20

sample_size = int(sample_rate*duration)
silent_thresh = 0.0005

def envelope(signal, rate, thresh):
    mask = []
    y = pd.Series(signal).apply(np.abs)
    # Create aggregated mean
    y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean()
    for m in y_mean:
        mask.append(m > thresh)

    return mask

def load_audio(path):
    signal, rate = librosa.load(path, sr=sample_rate)
    mask = envelope(signal, rate, silent_thresh)
    signal = signal[mask]

    return signal

def melspectrogram(signal):
    spectro = librosa.feature.melspectrogram(
        signal,
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft
    )
    spectro = librosa.power_to_db(spectro)
    spectro = spectro.astype(np.float32)
    return spectro

def minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

def plot_spectro(spectro):
    librosa.display.specshow(
        spectro,
        x_axis='time',
        y_axis='mel',
        sr=sample_rate,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title("Log-frequency power spectrogram")
    plt.show()

def audio_to_image(audio, out):
    signal = load_audio(audio)
    mels = melspectrogram(signal)
    skimage.io.imsave(out, mels)

def process(audio, out):
    signal = load_audio(audio)

    current = 0
    end = False
    count = 0

    if len(signal) < sample_size:
        return

    while not end:
        if current+sample_size > len(signal):
            sample = signal[len(signal)-sample_size:]
            end = True
        else:
            sample = signal[current:current+sample_size]
            current += sample_size

        mels = melspectrogram(sample)
        skimage.io.imsave(out+str(count)+".png", mels, check_contrast=False)
        count += 1

for l in LABELS:
    print("Processing: "+l)
    count = 0
    for audio in tqdm(os.listdir(DATA_FOLDER+l)):
        if os.path.splitext(audio)[-1] != ".wav":
            continue
        process(DATA_FOLDER+l+"/"+audio, OUT+l+"/"+str(count)+"-")
        count += 1

DATA_SIZE = len(os.listdir(OUT+"cough/"))

data = [[], []]
for i, label in enumerate(LABELS):
    samples = os.listdir(OUT+label)
    np.random.shuffle(samples)
    samples = samples[:DATA_SIZE]
    print("Processing: "+label)
    for image in tqdm(samples):
        img = cv2.imread(OUT+label+"/"+image)
        data[0].append(img)
        data[1].append(i)

np.save(DATASET, data)
