from __future__ import unicode_literals

import os
import math
import json
import copy
import youtube_dl
from config import *
from tqdm import tqdm
from pydub import AudioSegment
import librosa, librosa.display
from pydub.silence import split_on_silence

"""
This script downloads all the youtube links in the files named after the labels, then:
- Converts them into WAV
- Splits them into non silent chunks
- Splits the chunks into segments (the actual size of training data, so that we have same lenghts)
- Extracts the mfccs out of the segments
- Saves the data in a json file

(I split the data fetching at conversion to the mfcc because it was taking too much memory to do it all together)

Dataset from: https://www.karger.com/Article/FullText/504666
"""

os.system("clear")

pbar = None
queue = None

youtube_par = {
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '192',
    }],
    'quiet':True
}

data = {
    "mfcc":[],
    "labels":[]
}


def single(link, out, silence):
    conf = copy.deepcopy(youtube_par)
    conf["outtmpl"] = out+"%(title)s.mp3"
    try:
        f = ""
        # Download the video
        with youtube_dl.YoutubeDL(conf) as you:
            info = you.extract_info(link, download=True)
            f = out+""+info.get("title", None)
        pbar.update(0.25)
        # Convert it into WAV, needed because youtube_dl currupts wav files on dowload for some reason
        os.system("ffmpeg -i '" + f + ".mp3' '" + f + ".wav' >/dev/null 2>&1")
        os.system("rm '" + f + ".mp3'")
        # This doesn't run for non cough data because we want it to be noisy
        if silence:
            # Load whole wav file
            signal = AudioSegment.from_wav(f +".wav")
            # Split the wav file into chunks of non silent
            chunks = split_on_silence(signal, min_silence_len=500, silence_thresh=-35)
            pbar.update(0.25)
            # Exports all the chunks
            for i, c in enumerate(chunks):
                c.export(f + str(i) + ".wav", format="wav")
            os.system("rm '"+f+".wav'")
        else:
            pbar.update(0.25)

    except Exception as e:
        pass

    pbar.update(0.5)


def process(folder, label):
    items = os.listdir(folder)
    if label == 1:
        # To make sure we get same amount of data for both classes
        each = len(data["labels"])//len(items)
    for sample in tqdm(items):
        if os.path.splitext(sample)[-1] != ".wav":
            continue
        # Load the chunk into librosa
        signal, sr = librosa.load(folder+sample, sr=SR)
        # Decide the segments based on length
        segments = len(signal) // SEGMENT_LENGTH
        curr = 0  # For segment indexing
        # print(sample, segments)
        count = 0
        for segment in range(segments):
            if label == 1 and count >= each:
                break
            # Extract mfcc data
            mfcc = librosa.feature.mfcc(signal[curr:curr + SEGMENT_LENGTH], sr=SR, n_mfcc=N_MFCC, n_fft=N_FURIER,
                                        hop_length=HOP_LENGTH).T
            if len(mfcc) == EXPECTED_MFCC:
                data["mfcc"].append(mfcc.tolist())
                data["labels"].append(label)
                count += 1

            curr += SEGMENT_LENGTH

if __name__ == "__main__":
    FETCH = False

    try:
        os.system("mkdir data >/dev/null 2>&1")
    except:
        pass

    for i, label in enumerate(LABELS):
        try:
            os.system("mkdir data/"+label+" >/dev/null 2>&1")
        except:
            pass

        if FETCH:
            print("Processing label: "+label)
            with open(label+".txt", "r") as f:
                links = f.read().splitlines()

            pbar = tqdm(total=len(links))
            for l in links:
                if "youtube" in l:
                    single(l, "data/"+label+"/", i == 0)
        else:
            process("data/"+label+"/", i)

    if not FETCH:
        # Dump the dat into json file
        print(len(data["labels"]))
        with open(OUTPUT, "w") as j:
            json.dump(data, j, indent=4)