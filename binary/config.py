import math

DATA_FOLDER = "edited_wavs/"
OUTPUT = "dataset.json"
LABELS = ["cough", "not"]
SR = 16000
N_MFCC = 15
N_FURIER = 2048
HOP_LENGTH = 512
SEGMENT_LENGTH = SR//2
EXPECTED_MFCC = math.ceil(SEGMENT_LENGTH/HOP_LENGTH)