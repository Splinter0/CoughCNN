# CoughCNN

## Covid Anomaly detection model

Due to the lack of enough covid cough samples, and of the precise features in those coughs which classify it as covid;
we cannot use a simple supervised learning approach. Instead the approach taken currently is an unsupervised (which will
turn to semisupervised later using the avaiable data to refine the model) using an autoencoder. The autoencoder takes all
the non-covid cough samples which have been detected by the model in `spectro` and uses them as a training dataset, the autoencoder
learns the representation of a "normal" cough and how to recreate it (boils down the sample using convolutions and then re-builds
the sample using a transposition of the convolution). When a cough that was not similar to the ones which were in the dataset 
(which with enough data should be singled out to the covid coughs) the error which the model will make at recreating the sample
provided (calculate useing MSE of the original image with the one created by the model) will be high and the sample will be
labeled as an anomaly.

### Current

Under the folder `covid` there are two files:
**model.py**, contains the code for the autoencoder and the MSE applied to the melspectrogram, this still does not work unfortunately
I'm tweaking the input shape of the samples by increasing the sample length and the number of mels per sample but I'm still getting
problems. 
**processing.ipynb** contains the data processing to create the dataset used

**short-spectro** is a clone folder of `spectro` but contains the scripts to make the data suitable for the anomaly detection model,
once a stable solution for the anomaly detection will be finished all the samples will be in the same format and usable in any model. 
As of now I'll be keeping both separate.

Here is all the data processed by the short-spectro: https://drive.google.com/file/d/1aAMGHTFJjiv7K6DrN_1DptcCeA3Efmbk/view?usp=sharing

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
https://app.wandb.ai/mastersplinter/CoughDetect/sweeps/phdtst8z

### Next

- Data processing hyperparameters
- Testing different sample length and optimize them

### Resources that inspired this:

- https://arxiv.org/pdf/2004.01275.pdf (for initial model architecture)
- https://www.mi.t.u-tokyo.ac.jp/assets/publication/LEARNING_ENVIRONMENTAL_SOUNDS_WITH_END-TO-END_CONVOLUTIONAL_NEURAL_NETWORK.pdf
- https://www.cs.tut.fi/~tuomasv/papers/ijcnn_paper_valenti_extended.pdf
- https://adventuresinmachinelearning.com/global-average-pooling-convolutional-neural-networks/
- https://arxiv.org/pdf/1809.04437.pdf
- https://arxiv.org/pdf/1711.10282.pdf
