# CoughCNN
A new approach using MFCC data and a CNN to classify dry/wet cough from audio

## Binary how to run

- `model.py` contains the model to in keras to train (model isn't saved yet)
- `processing.py` contains the functions to fetch and save the dataset, and to extract and save the mfcc, toggle variable `FETCH` manually to choose if you want to fetch or process the dataset
- resize.py contains functions to make the downloaded dataset smaller, some of the videos are huge and take too much memory.
