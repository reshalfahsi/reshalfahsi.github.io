---
layout: page 
title: Software 
---


## Biomedical Engineering and Life Sciences ##

### EEG Motor Imagery Classification Using CNN, Transformer, and MLP

<div align="center">

<img src="https://raw.githubusercontent.com/reshalfahsi/eeg-motor-imagery-classification/master/assets/cnn-transformer-mlp-white.png" width="600">

An illustration of the CNN-Transformer-MLP model.
</div>

The electroencephalogram, or EEG for short, is one of the biosignals that display brain activity in the form of time-series data. EEG can be used to help amputees or paralyzed people move their prosthetic arms via a brain-computer interface (BCI). In order to identify the correct limbs to control from the EEG signal, a combination of CNN, Transformer, and MLP is utilized in this work for motor imagery (MI) classification. CNN converts the epoched EEG signal into meaningful representation in accordance with the signal's non-stationary nature. Transformer finds the global relationship of the given representation from CNN. MLP classifies the expected upper limbs to move based on the extracted information from the Transformer. To gauge the capability of the CNN-Transformer-MLP model, PhysioNet's EEG Motor Movement/Imagery Dataset is used. The model attains an accuracy of ``76.4%`` on the test set. This project's source code is hosted on [Github](https://github.com/reshalfahsi/eeg-motor-imagery-classification).


### COVID19CT3D

<div align="center">

<img src="{{site.baseurl}}public/thorax.gif" width=300 style="float:right margin-left=10cm">

</div>

This tutorial will teach you how to train a Deep Learning model based on 3D Convolution. This model will classify whether the volumetric medical image from a 3D CT scan of the thorax is infected by COVID-19 or not. The model's output is a single-valued tensor that represents the probability of being infected by COVID-19. This tutorial is based on [A tutorial notebook on 3D image classification](https://github.com/hasibzunair/3D-image-classification-tutorial). This project's source code is hosted on [Github](https://github.com/reshalfahsi/covid19ct3d).


### GGB

<div align="center">

<img src="{{site.baseurl}}public/GGB_RGB_LEUKOCYTES.jpg" width=400 style="float:right margin-left=10cm">

</div>

This package is implementation of GGB color space from [Development of a Robust Algorithm for Detection of Nuclei and Classification of White Blood Cells in Peripheral Blood Smear Image](https://link.springer.com/content/pdf/10.1007%2Fs10916-018-0962-1.pdf). GGB's source code is hosted on [Github](https://github.com/reshalfahsi/ggb).


### MyQLaNet

<div align="center">

<img src="{{site.baseurl}}public/myqlanet.jpg" width=400 style="float:right margin-left=10cm">

</div>

MyQLaNet is a Deep Learning platform for macula detection. It provides end to end system for macula detection with graphical user interface. MyQLaNet's source code is hosted on [Github](https://github.com/reshalfahsi/myqlanet).


-----


## Computer Vision ##


### Stable Diffusion Dreaming

<div align="center">
    <a href="https://youtu.be/OBymeX0mtCE">
        <img src="https://github.com/reshalfahsi/stable-diffusion-dreaming-notebook/blob/main/assets/stablediffusion.gif?raw=true" width=400 />
    </a>

Stable diffusion dreams of "Alien invasion of Mars colonization in the future".
</div>

Generate video by stable diffusion in Colab Notebook. This project's source code is hosted on [Github](https://github.com/reshalfahsi/stable-diffusion-dreaming-notebook).


### Image Captioning API

<div align="center">

<img src="{{site.baseurl}}public/image-captioning.gif" width="400">

</div>

Minimal implementation of image captioning API hosted on [Heroku](https://image-captioning-69420.herokuapp.com/). It receives an image and responds with a caption regarding the image. This project's source code is hosted on [Github](https://github.com/reshalfahsi/image-captioning-api).


### SotoDemoBot

<div align="center">

<img src="{{site.baseurl}}public/sotodemobot.gif" width="400">

</div>

Simple Object detection Telegram bOt DEMO: predict the objects in the given image. Use ``/predict <URL>`` to predict the objects in the image of given url. This project's source code is hosted on [Github](https://github.com/reshalfahsi/object-detection-bot).


-----



## Natural Language Processing ##


### Movie Review Sentiment Analysis Using CNN and MLP

<div align="center">

<img src="https://raw.githubusercontent.com/reshalfahsi/movie-review-sentiment-analysis/master/assets/neg_CNN0.png" width="600">

Visualization of the first layer of CNN on the negative review.
</div>

Audiences' reactions to the movie they have watched can be presented in a text format called reviews. These reviews can be polarized into two clusters: positive responses and negative responses. Using CNN and MLP, one can perform sentiment analysis on movie reviews to automatically recognize the viewer's tendency toward a particular movie. CNN is used for extracting the latent information within the text format. MLP leverages the extracted information and carries out the classification task. The CNN-MLP model is evaluated with Standford's IMBD Movie Review dataset. On the test set, the model achieves ``85.6%`` accuracy. This project's source code is hosted on [Github](https://github.com/reshalfahsi/movie-review-sentiment-analysis).


### Your Kind Friend Bot: Unreliable Chatbot

<div align="center">

<img src="{{site.baseurl}}public/ykfbot.gif" width="400">

</div>

Your typical kind friend who talk nonsense just to kill time. It can respond to text or image. Visit [here](https://t.me/yourkindfriendbot). This project's source code is hosted on [Github](https://github.com/reshalfahsi/ykfbot).


-----


## Audio Data ##

### Music Genre Classification

<div align="center">

<img src="https://raw.githubusercontent.com/reshalfahsi/music-genre-classification/master/classification.png" width="600">

</div>

Classify input audio into a particular genre of music. First, the audio is preprocessed via [MFCC](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum). Next, using MLP, we obtain the probability distribution of 10 classes of music genres. Before applying MLP to the MFCC, the cepstral coefficients with the length of the number of sequence of time has to be averaged and subjected to [CMVN](https://en.wikipedia.org/wiki/Cepstral_mean_and_variance_normalization). This project's source code is hosted on [Github](https://github.com/reshalfahsi/music-genre-classification).


### AI Cover Song

<div align="center">
    <a href="https://youtu.be/wU6WnPl54HI">
        <img src="https://img.youtube.com/vi/wU6WnPl54HI/hqdefault.jpg" alt="Itsuki Nakano - Asmalibrasi (AI Cover)" width=400 />
    </a>

Itsuki Nakano - Asmalibrasi (AI Cover).
</div>

Cover your favorite song by your favorite singer. This project's source code is hosted on [Github](https://github.com/reshalfahsi/AI-Cover-Song).


-----


## Deep Learning ##


### Neural Network

<div align="center">

<img src="https://4.bp.blogspot.com/-Anllqq6pDXw/VRUSesbvyAI/AAAAAAAAsrc/CIHz6vLsuTU/s800/computer_jinkou_chinou.png" width="300">

</div>

A naive implementation of neural network. The code structure is heavily inspired by [PyTorch](https://github.com/pytorch/pytorch) and [TensorFlow](https://github.com/tensorflow/tensorflow). However, this package is used for educational purposes and is not intended to be adopted in production. This project's source code is hosted on [Github](https://github.com/reshalfahsi/neuralnetwork).


### PyTorch Depthwise Separable Convolution

<div align="center">

<img src="https://production-media.paperswithcode.com/methods/Screen_Shot_2020-05-31_at_10.30.20_PM.png" width="400">

</div>

PyTorch (unofficial) implementation of Depthwise Separable Convolution. This type of convolution is introduced by Chollet in [Xception: Deep Learning With Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357). This package provides ``SeparableConv1d``, ``SeparableConv2d``, ``SeparableConv3d``, ``LazySeparableConv1d``, ``LazySeparableConv2d``, and ``LazySeparableConv3d``. This package's source code is hosted on [Github](https://github.com/reshalfahsi/separableconv-torch).



-----


## Robotics ##


### Suction Arm Manipulator Robot

<div align="center">
    <a href="https://youtu.be/cmVsOR96NVk">
        <img src="https://github.com/reshalfahsi/arm-suction-sim/blob/master/img/simulation.gif?raw=true" width=400 />
    </a>
</div>

Simulate the Suction Arm Manipulator Robot to pickup daily objects inspired by Amazon Robotics Challenge. This project's source code is hosted on [Github](https://github.com/reshalfahsi/arm-suction-sim).


-----


## Other Open Source Software ##

For a list of my open source software, please take a look at my [Github](https://github.com/reshalfahsi).
