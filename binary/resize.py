import os
from tqdm import tqdm
from pydub import AudioSegment

"""
This is a helper script to chop up large audio files
"""

MAX=2*60*1000
FOLDER = "data/not/"
SKIP = 6
N_PER = 5

for sample in tqdm(os.listdir(FOLDER)):
    n = os.path.splitext(sample)
    if n[-1] != ".wav":
        continue

    signal = AudioSegment.from_wav(FOLDER+sample)
    curr = 0
    for i in range(N_PER):
        if i+1 % SKIP == 0:
            continue
        signal[curr:curr+MAX].export(FOLDER+n[0]+str(i)+".wav", format="wav")
        curr += MAX

    os.system("rm '"+FOLDER+sample+"'")