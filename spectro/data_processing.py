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

DATA_FOLDER = "data/"
OUT = "samples/"
DATASET = "dataset.npy"
LABELS = ["cough", "not"]

musan = os.listdir(DATA_FOLDER+"musan/")
NOISES = {}
for category in musan:
    if not os.path.isdir(DATA_FOLDER+"musan/"+category):
        continue

    NOISES[category] = []
    for s in os.listdir(DATA_FOLDER+"musan/"+category):
        if os.path.splitext(s)[-1] != ".wav":
            continue

        NOISES[category].append(DATA_FOLDER+"musan/"+category+"/"+s)

MIX_THRESH = 0.25

sample_rate = 44000
duration = 0.5 #seconds
durs = int(np.ceil(sample_rate*duration))
hop_length = 347*duration
n_mels = 128
fmin = 20
fmax = sample_rate//2
n_fft = n_mels*20

sample_size = int(sample_rate*duration)
silent_thresh = 0.0005

global TASTE
TASTE = True

def envelope(signal, rate, thresh):
    mask = []
    y = pd.Series(signal).apply(np.abs)
    # Create aggregated mean
    y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean()
    for m in y_mean:
        mask.append(m > thresh)

    return mask

def load_audio(path, silence=True):
    signal, rate = librosa.load(path, sr=sample_rate)
    if silence:
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

def manual_mix(sample, noise, ratio):
    gap = len(noise)-len(sample)
    point = 0
    if gap > 0:
        point = np.random.randint(low=0, high=len(noise)-len(sample))
    noise = noise[point:point+len(sample)]
    final = []
    for f in range(len(sample)):
        n = noise[f]*ratio
        final.append(sample[f]+n)

    return np.array(final)

def augment(signal, annoy):
    s = len(signal)
    signals = []
    for a in annoy:
        signals.append(manual_mix(signal, a, MIX_THRESH))

    return signals

def get_rand(k):
    while True:
        noise = load_audio(NOISES[k][np.random.randint(low=0, high=len(NOISES[k]))], silence=False)
        if len(noise) < durs:
            continue
        return noise

def process(audio, out, a=False):
    global TASTE
    signal = load_audio(audio)

    if len(signal) < sample_size:
        return

    if a:
        annoy = [
            get_rand("noise"),
        ]
        r = np.random.randint(2)
        e = "music" if r == 0 else "speech"
        annoy.append(get_rand(e))

    current = 0
    end = False
    count = 0

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

        if a:
            signals = augment(sample, annoy)
            for i, s in enumerate(signals):
                mels = melspectrogram(s)
                skimage.io.imsave(out+str(count)+"-"+str(i)+".png", mels, check_contrast=False)

            if TASTE:
                librosa.output.write_wav("gigataste1.wav", signals[0], sample_rate)
                librosa.output.write_wav("gigataste2.wav", signals[1], sample_rate)
                librosa.output.write_wav("origintaste.wav", sample, sample_rate)
                TASTE = False

def gigasmurf(DATA_FOLDER, OUT, DATASET, aug=True):
    print("Processing folder: "+DATA_FOLDER)
    for l in LABELS:
        print("Processing: "+l)
        count = 0
        samples = os.listdir(DATA_FOLDER+l)
        np.random.shuffle(samples)
        for audio in tqdm(samples):
            if os.path.splitext(audio)[-1] != ".wav":
                continue
            process(DATA_FOLDER+l+"/"+audio, OUT+l+"/"+str(count)+"-", a=aug and l == "cough")
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


gigasmurf(DATA_FOLDER, OUT, DATASET)
gigasmurf("test/", "test/samples/", "test/test.npy", aug=False)
