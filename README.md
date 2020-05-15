# CoughCNN

## Log Melspectrogram

Under the folder `spectro` you can find the whole approach using the melspectrograms to train a Convolutional Neural Network

### Current

Right now the data that has been used comes from a cough dataset from: https://www.karger.com/Article/FullText/504666
and using manually labeled data coming from our webapp. The testing dataset is also manyally labeled data coming
from our webapp but that obviosly the CNN has not seen before.

The current accuracy on testing data ranges between 68-75% which isn't great yet. The training accuracy reaches 97%.

This was acheived augmenting the dataset by mixing the cough sample with some background noise to make them
more real world. It was used a mixing ration of 0.25 (0.25 of the noise signal added) using the musan dataset
https://www.openslr.org/17/

The samples are now a grayscale melspectrogram of 0.5s.

All the data of the project can be found at : https://drive.google.com/drive/folders/1deqYCDye5l95RGJCeKXlcqH9Ras7lRQr?usp=sharing

### Next

- Look into other augmenting techniques
- Try out 1s sampling
- VOCO dataset
- Different CNN structure and hyperparameters
