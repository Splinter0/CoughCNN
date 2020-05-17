# CoughCNN

## Log Melspectrogram

Under the folder `spectro` you can find the whole approach using the melspectrograms to train a Convolutional Neural Network

### Current

Right now the data that has been used comes from a cough dataset from: https://www.karger.com/Article/FullText/504666
and using manually labeled data coming from our webapp. The testing dataset is also manyally labeled data coming
from our webapp but that obviosly the CNN has not seen before.

This was acheived augmenting the dataset by mixing the cough sample with some background noise to make them
more real world. It was used a mixing ration of 0.25 (0.25 of the noise signal added) using the musan dataset
https://www.openslr.org/17/

The samples are now a grayscale melspectrogram of 0.5s.

All the data of the project can be found at : https://drive.google.com/drive/folders/1deqYCDye5l95RGJCeKXlcqH9Ras7lRQr?usp=sharing

Right now we have normalized the data and we use a new model structure using GlobalAveragePooling2D which gave a huge performance
boost to the previous version. We went from 68-75% to 80-84% accuracy on our test set created from the webapp.
Right now we are running hyperparameters optimization and you can check out everything at:
https://app.wandb.ai/mastersplinter/CoughDetect

### Next

- Data processing hyperparameters
