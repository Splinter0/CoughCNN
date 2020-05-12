# CoughCNN
A new approach using MFCC data and a CNN to classify dry/wet cough from audio

## Binary how to run

- `model.py` contains the model to in keras to train (model isn't saved yet)
- `processing.py` contains the functions to fetch and save the dataset, and to extract and save the mfcc, toggle variable `FETCH` manually to choose if you want to fetch or process the dataset
- `resize.py` contains functions to make the downloaded dataset smaller, some of the videos are huge and take too much memory.
- `detector.py` uses the model to classify an audio file

### WanDB

Used WanDB for hyperparameters optimization, the whole project is here : https://app.wandb.ai/mastersplinter/CoughDetection-hyper

## Log Mel Spectrogram Approach
- Data processing notebook 
- Hyper parameter optimization using some data from webapp dataset under "real_acc" : https://app.wandb.ai/mastersplinter/MelSpect-CoughDetect
