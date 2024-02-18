---
layout: page 
title: Software 
---


## Biomedical Engineering and Life Sciences ##


### Medical Image Similarity Search Using a Siamese Network With a Contrastive Loss ###

<div align="center">

<img src="https://raw.githubusercontent.com/reshalfahsi/medical-image-similarity-search/master/assets/qualitative.png" width="600">

The image similarity search results for DermaMNIST (first row), PneumoniaMNIST (second row), RetinaMNIST (third row), and BreastMNIST (fourth row).
</div>

Obtaining the ontological account of an image numerically can be earned via a Siamese network. The anatomy of this network has a twin architecture, consisting of convolutional and fully connected layers with shared weights. Each architecture digests an image and yields the vector embedding (the ontological or latent representation) of that image. These two vectors are then subjected to the Euclidean distance calculation. Next, the result is funneled to the last fully connected layer to get the logit describing their similarity. To learn the representation, here, we can leverage contrastive loss as our objective function to be optimized. The network is trained on paired images, i.e., positive and negative. In this project, the positive pairs are two images that belong to the same dataset, and the negative pairs are two images from distinct datasets. Here, subsets of the MedMNIST dataset are utilized: DermaMNIST, PneumoniaMNIST, RetinaMNIST, and BreastMNIST. Then, accuracy is used to evaluate the trained network. Afterward, we encode all images of the train and validation sets into embedding vectors and store them in the PostgreSQL database. So, sometimes later, we can use the embedding vectors to retrieve similar images based on a query image (we can obtain it from the test set). To find similar images, FAISS (Facebook AI Similarity Search) is employed. FAISS helps us seek the closest vectors to the query vector. This project's source code is hosted on [GitHub](https://github.com/reshalfahsi/medical-image-similarity-search).



### Knowledge Distillation for Skin Lesion Classification ###

<div align="center">

<img src="https://raw.githubusercontent.com/reshalfahsi/knowledge-distillation/master/assets/distilled_qualitative.png" width="600">

The qualitative result of the distilled model.
</div>

The goal of knowledge distillation is to improve the performance of the half-witted model, which, most of the time, has fewer parameters, by allowing it to learn from the more competent model or the teacher model. The half-witted model, or the student model, excerpts the knowledge from the teacher model by matching its class distribution to the teacher model's. To make the distributions softer (used in the training process as the part of the loss function), we can adjust a temperature _T_ to them (this is done by dividing the logits before softmax by the temperature). This project designates EfficientNet-B0 as the teacher and SqueezeNet v1.1 as the student. These models will be experimented on the DermaMNIST dataset of MedMNIST. We will take a look at the performance of the teacher, the student (without knowledge distillation), and the student (with knowledge distillation) in the result section. This project's source code is hosted on [GitHub](https://github.com/reshalfahsi/knowledge-distillation).



### Medical Image Latent Space Visualization Using VQ-VAE ###

<div align="center">

<img src="https://raw.githubusercontent.com/reshalfahsi/medical-image-latent-space-visualization/master/assets/latent_space.png" width="600">

The latent space of five distinct datasets, i.e., DermaMNSIT, PneumoniaMNIST, RetinaMNIST, BreastMNIST, and BloodMNIST.
</div>

In this project, VQ-VAE (Vector Quantized VAE) is leveraged to learn the latent representation _z_ of various medical image datasets _x_ from MedMNIST. Similar to VAE (Variational Autoencoder), VQ-VAE consists of an encoder _q_(_z_<code>&#124;</code>_x_) and a decoder _p_(_x_<code>&#124;</code>_z_). But unlike VAE, which generally uses the Gaussian reparameterization trick, VQ-VAE utilizes vector quantization to sample the latent representation _z_ ~ _q_(_z_<code>&#124;</code>_x_). Using vector quantization, it allows VQ-VAE to replace a generated latent variable from the encoder with a learned embedding from a codebook __C__ ∈ R<sup>_E_ × _D_</sup>, where E is the number of embeddings and _D_ is the number of latent variable dimensions (or channels in the context of image data). Let __X__ ∈ R<sup>_H_ × _W_ × _D_</sup> be the output feature map of the encoder, where _H_ is the height and _W_ is the width. To transform the raw latent variable to the discretized one, first we need to find the Euclidean distance between __X__ and __C__. This step is essential to determine the closest representation of the raw latent variable to the embedding. The computation of this step is roughly expressed as: (__X__)<sup>2</sup> + (__C__)<sup>2</sup> - 2 × (__X__ × __C__). This calculation yields __Z__ ∈ R<sup>_H_ × _W_</sup>, where each element denotes the index of the nearest embedding of the corresponding latent variable. Then, __Z__ is subject to __C__ to get the final discrete representation. Inspired by the centroid update of K-means clustering, EMA (exponential moving average) is applied during training, which updates in an online fashion involving embeddings in the codebook and the estimated number of members in a cluster. This project's source code is hosted on [GitHub](https://github.com/reshalfahsi/medical-image-latent-space-visualization).


### Medical Image Generation Using Diffusion Model ###

<div align="center">

<img src="https://raw.githubusercontent.com/reshalfahsi/medical-image-generation/master/assets/qualitative_result.png" width="600">

Unconditional progressive generation on the BloodMNIST dataset (left) and a montage of the actual BloodMNIST dataset (right).
</div>

Image synthesis on medical images can aid in generating more data for biomedical problems, which is hindered due to some legal and technical issues. Using the diffusion model, this problem can be solved. The diffusion model works by progressively adding noise, typically Gaussian, to an image until it is entirely undistinguishable from randomly generated pixels. Then, the noisy image is restored to its original appearance gradually. The forward process (noise addition) is guided by a noise scheduler, and the backward process (image restoration) is carried out by a U-Net model. In this project, the diffusion model is trained on the BloodMNIST dataset from the MedMNIST dataset. This project's source code is hosted on [GitHub](https://github.com/reshalfahsi/medical-image-generation).


### Small Molecular Graph Generation for Drug Discovery ###


<div align="center">

<img src="https://raw.githubusercontent.com/reshalfahsi/molecule-generation-drug-discovery/master/assets/qualitative_result.png" width="600">

The qualitative results of the generated molecules. The chemical structure, the SMILES representation, and the QED scores are provided.
</div>


With the advent of deep learning, [drug development](https://en.wikipedia.org/wiki/Drug_development) can be sped up just by learning the patterns within the molecules regarding their chemical properties and composition. The pursuit of good candidates for drugs can be achieved using the generative model which can extrapolate the unseen molecular structure. In this project, one of the most popular generative models, ``Generative Adversarial Network`` or ``GAN``, is utilized. The generator of GAN consists of MLP, and the discriminator of GAN consists of R-GCN + MLP. Nowadays, there are plenty of open-sourced datasets that can be used for this purpose such as the ``QM9 (Quantum Machines 9) dataset``. The GAN model is trained on QM9 dataset and its performances are assessed by means of [molecular metrics](https://github.com/nicola-decao/MolGAN/blob/master/utils/molecular_metrics.py), i.e., quantitative estimate of druglikeness (QED), solubility (defined as the log octanol-water partition coefficient or logP), synthetizability, natural product, drug candidate, valid, unique, novel, and diversity. This project's source code is hosted on [GitHub](https://github.com/reshalfahsi/molecule-generation-drug-discovery).


### Self-Supervised Contrastive Learning for Colon Pathology Classification ###


<div align="center">

<img src="https://raw.githubusercontent.com/reshalfahsi/contrastive-ssl-pathology/master/assets/fine-tuned_qualitative.png" width="600">

The qualitative result of the fine-tuned pre-trained SSL model.
</div>


Self-supervised learning, or SSL, has become a modern way to learn the hidden representation of data points. A dataset is not always provided with a label that marks a data point's category or value. SSL mitigates this issue by projecting a data point into an embedding vector representing information beneath. SSL can be trained contrastively, i.e., to measure the similarity between two projected embeddings (original and augmented) using certain metrics, e.g., cosine similarity, Euclidean distance, Manhattan distance, etc. By learning the latent representation, the SSL model can be utilized as a pre-trained model and fine-tuned as needed. The SSL model is divided into three parts: the backbone feature extractor, the embedding projection head, and the classification head. The backbone feature extractor leverages ResNet 18. The embedding head gives the embedding vector. The classification head concludes the classification task's result. Here, two other models are also introduced: the baseline model and the fine-tuned pre-trained SSL model. Both of them consist of a backbone feature extractor and a classification head. Yet, the latter makes use of the trained SSL model's backbone as its own backbone. To evaluate the performance of the models, the PathMNIST of the MedMNIST dataset is utilized. On batched training, the other pairs in the batch relative to a certain pair (positive pair) are treated as negative pairs. This notion is useful for the computation of the contrastive loss: NTXentLoss/InfoNCE. This project's source code is hosted on [GitHub](https://github.com/reshalfahsi/contrastive-ssl-pathology).


### EEG Motor Imagery Classification Using CNN, Transformer, and MLP

<div align="center">

<img src="https://raw.githubusercontent.com/reshalfahsi/eeg-motor-imagery-classification/master/assets/cnn-transformer-mlp-white.png" width="600">

An illustration of the CNN-Transformer-MLP model.
</div>

The electroencephalogram, or EEG for short, is one of the biosignals that display brain activity in the form of time-series data. EEG can be used to help amputees or paralyzed people move their prosthetic arms via a brain-computer interface (BCI). In order to identify the correct limbs to control from the EEG signal, a combination of CNN, Transformer, and MLP is utilized in this work for motor imagery (MI) classification. CNN converts the epoched EEG signal into meaningful representation in accordance with the signal's non-stationary nature. Transformer finds the global relationship of the given representation from CNN. MLP classifies the expected upper limbs to move based on the extracted information from the Transformer. To gauge the capability of the CNN-Transformer-MLP model, PhysioNet's EEG Motor Movement/Imagery Dataset is used. The model attains an accuracy of ``76.4%`` on the test set. This project's source code is hosted on [GitHub](https://github.com/reshalfahsi/eeg-motor-imagery-classification).


### COVID19CT3D

<div align="center">

<img src="{{site.baseurl}}public/thorax.gif" width=300 style="float:right margin-left=10cm">

</div>

This tutorial will teach you how to train a Deep Learning model based on 3D Convolution. This model will classify whether the volumetric medical image from a 3D CT scan of the thorax is infected by COVID-19 or not. The model's output is a single-valued tensor that represents the probability of being infected by COVID-19. This tutorial is based on [A tutorial notebook on 3D image classification](https://github.com/hasibzunair/3D-image-classification-tutorial). This project's source code is hosted on [GitHub](https://github.com/reshalfahsi/covid19ct3d).


### GGB

<div align="center">

<img src="{{site.baseurl}}public/GGB_RGB_LEUKOCYTES.jpg" width=400 style="float:right margin-left=10cm">

</div>

This package is implementation of GGB color space from [Development of a Robust Algorithm for Detection of Nuclei and Classification of White Blood Cells in Peripheral Blood Smear Image](https://link.springer.com/content/pdf/10.1007%2Fs10916-018-0962-1.pdf). GGB's source code is hosted on [GitHub](https://github.com/reshalfahsi/ggb).


### MyQLaNet

<div align="center">

<img src="{{site.baseurl}}public/myqlanet.jpg" width=400 style="float:right margin-left=10cm">

</div>

MyQLaNet is a Deep Learning platform for macula detection. It provides end to end system for macula detection with graphical user interface. MyQLaNet's source code is hosted on [GitHub](https://github.com/reshalfahsi/myqlanet).


-----


## Computer Vision ##

### Next-Frame Prediction Using Convolutional LSTM ###

<p align="center"> 
  <img src="https://raw.githubusercontent.com/reshalfahsi/next-frame-prediction/master/assets/result.gif" alt="qualitative" > 
  <br /> The Convolutional LSTM model predicts the ensuing frame-by-frame from <i>t</i> = 1 to <i>t</i> = 19.
</p>

In the next-frame prediction problem, we strive to generate the subsequent frame of a given video. Inherently, video has two kinds of information to take into account, i.e., image (spatial) and temporal. Using the Convolutional LSTM model, we can manage to feature-extract and process both pieces of information with their inductive biases. In Convolutional LSTM, instead of utilizing fully connected layers within the LSTM cell, convolution operations are adopted. To evaluate the model, the moving MNIST dataset is used. To evalute the model, the Moving MNIST dataset is used. This project's source code is hosted on [GitHub](https://github.com/reshalfahsi/next-frame-prediction).



### Point Cloud Segmentation Using PointNet ###


<p align="center"> 
   <a href="https://reshalfahsi.github.io/point-cloud-segmentation">
       <img src="https://raw.githubusercontent.com/reshalfahsi/point-cloud-segmentation/master/assets/result.png" alt="qualitative_result"> 
   </a>
   The segmentation result for the motorbike subcategory of the ShapeNet dataset with the labels: <i>wheel</i>, <i>seat</i>, <i>gas_tank</i>, <i>light</i>, and <i>handle</i>.
</p>


In this project, PointNet is leveraged for the segmentation of parts of a certain shape in the form of point cloud data. The data points are obtained from the ShapeNet dataset, i.e., ShapeNetCore. This project chooses the shape of a motorbike. PointNet is utilized due to its nature, which is invariant to permutation. Keep in mind that point cloud data has zero care for the spatial relationship between points in the point cloud, even though it stores information regarding the object's location in the space. In other words, the order of points must be negligible and not influence the result. This project's source code is hosted on [GitHub](https://github.com/reshalfahsi/point-cloud-segmentation).



### Action Recognition Using CNN + Bidirectional RNN ###


<div align="center"> 

<img src="https://raw.githubusercontent.com/reshalfahsi/action-recognition/master/assets/result.gif" alt="qualitative_result" width="400"> 

The action recognition results of the CNN + Bidirectional RNN model. Several actions are shown in the compilation video: <i>brush hair</i>, <i>throw</i>, <i>dive</i>, <i>ride bike</i>, and <i>swing baseball.</i>

</div>


Given a video, we can undergo recognition or analysis to decide what action occurred in the clip. By nature, videos are a sequence of frames. Consequently, performing action recognition on video deals with processing spatio-temporal data. Here, we can make use of the HMDB51 dataset, consisting of 6k+ clips of 51 actions. This dataset has three separate train/test splits. Striving for simplicity, this project utilizes the first training split as the training set, the second testing split as the validation set, and the third testing split as the testing set. Regarding the action recognition model, CNN is customarily adopted to extract spatial information. Thus, a CNN architecture, MnasNet, is put into use. Next, to handle the temporal information, bidirectional RNN is employed. Succinctly, the action recognition model in this project is composed of CNN and bidirectional RNN. This project's source code is hosted on [GitHub](https://github.com/reshalfahsi/action-recognition).


### Novel View Synthesis Using NeRF ###

<div align="center"> 

<img src="https://raw.githubusercontent.com/reshalfahsi/novel-view-synthesis/master/assets/qualitative_result.gif" alt="qualitative_result" width="400"> 

The rendered 3D view of a bulldozer viewed from <i>x</i> = 0, <i>y</i> = 0, <i>z</i> = 3.5, <i>ϕ</i> = −15°, and <i>θ</i> = 0° to 360°. 

</div>

Legend has it that the artificial neural network (ANN) is infamously known as the universal approximator, which can fit any existing function. By exploiting this fact, we can build a network that approximates a function that maps spatial positions (_x_, _y_, _z_) and camera rays (these rays are acquired through the calculation of the camera matrix involving viewing directions (_θ_ (rotating along the _y_-axis), _ϕ_ (rotating along the _x_-axis)) and the spatial positions) to RGB pixels. Such a network, called the Neural Radiance Field, or NeRF in short, can be used to solve the problem of novel view synthesis of a scene. The network is coerced to overfit the function, which generates an RGB image (and also a depth map). These generated images (the final images are procured by computing the transmittance that is applied to the freshly generated images) from multiple angles are then collected, rendering the 3D representation of a certain object. In this project, a bulldozer from the Tiny NeRF dataset is used. This project's source code is hosted on [GitHub](https://github.com/reshalfahsi/novel-view-synthesis).


### Image Super-Resolution Using ESRGAN ###

<div align="center">

<img src="https://raw.githubusercontent.com/reshalfahsi/image-super-resolution/master/assets/qualitative_result.png" width="600">

Qualitative comparison between the reference high-resolution images (left column), high-resolution images via bicubic interpolation (middle column), and predicted high-resolution images through ESRGAN (right column).
</div>

Image super-resolution attempts to produce pixels within the image to fill the lack of information due to its low-resolution nature. Hence, it yields a higher-resolution image. One approach to this problem is via generative networks, e.g., ESRGAN (Enhanced Super-Resolution Generative Adversarial Network). This type of GAN is built explicitly for image super-resolution by considering several losses, i.e., contextual loss (focus on the distribution of the feature), perceptual loss (pixel-wise loss), and adversarial loss. These three losses are utilized for the generator loss. On the contrary, the discriminator loss only takes into account the adversarial loss. There are two stages during training: (1) train only the generator on the perceptual loss, and (2) train the generator and discriminator based on those, as mentioned earlier. The model is trained and evaluated on the BSDS500 dataset. The final result of the predicted high-resolution image is subjected to the sharpening method by subtracting the image with the Laplacian of the image. This project's source code is hosted on [GitHub](https://github.com/reshalfahsi/image-super-resolution).


### Zero-Reference Low-Light Image Enhancement ###

<div align="center">

<img src="https://raw.githubusercontent.com/reshalfahsi/zero-reference-low-light-image-enhancement/master/assets/enhancement_qualitative_00.png" width="600">
<img src="https://raw.githubusercontent.com/reshalfahsi/zero-reference-low-light-image-enhancement/master/assets/enhancement_qualitative_01.png" width="600">
<img src="https://raw.githubusercontent.com/reshalfahsi/zero-reference-low-light-image-enhancement/master/assets/enhancement_qualitative_02.png" width="600">

The qualitative results of the image enhancement method (comparing the original, the ground-truth, the PIL autocontrast, and the prediction).
</div>

Low-light image enhancement aims to raise the quality of pictures taken in dim lighting, resulting in a brighter, clearer, and more visually appealing image without adding too much noise or distortion. One of the state-of-the-art methods for this computer vision task is Zero-DCE. This method uses just a low-light image without any image reference to learn how to produce an image with higher brightness. There are four loss functions crafted specifically for this zero-reference low-light image enhancement method, i.e., color constancy loss, exposure loss, illumination smoothness loss, and spatial consistency loss. This project's source code is hosted on [GitHub](https://github.com/reshalfahsi/zero-reference-low-light-image-enhancement).


### Anchor-Free Object Detection

<div align="center">

<img src="https://raw.githubusercontent.com/reshalfahsi/anchor-free-object-detection/master/assets/qualitative_result.png" width="1200">

A person, a motorbike (left), a bird (middle), and two TV monitors (right) are detected.
</div>

Anchor boxes have been the prevalent way to generate candidates for the ground truth bounding boxes in the object detection problem. Yet, this approach is such a hassle and downright confusing. This tutorial leverages an object detection method named [FastestDet](https://github.com/dog-qiuqiu/FastestDet) that is lightweight and anchor-free. ``PASCAL VOC 2007 and 2012`` datasets are utilized to evaluate the model's capability. Here, the train and validation sets of ``PASCAL VOC 2012`` are used for the train, validation, and test set in this tutorial with a ratio of ``81:9:10``. Eventually, the inference set (the test set of ``PASCAL VOC 2007``) is used to see the qualitative performance of the model. This project's source code is hosted on [GitHub](https://github.com/reshalfahsi/anchor-free-object-detection).



### Stable Diffusion Dreaming

<div align="center">
    <a href="https://youtu.be/OBymeX0mtCE">
        <img src="https://github.com/reshalfahsi/stable-diffusion-dreaming-notebook/blob/main/assets/stablediffusion.gif?raw=true" width=400 />
    </a>

Stable diffusion dreams of "Alien invasion of Mars colonization in the future".
</div>

Generate video by stable diffusion in Colab Notebook. This project's source code is hosted on [GitHub](https://github.com/reshalfahsi/stable-diffusion-dreaming-notebook).


### Image Captioning API

<div align="center">

<img src="{{site.baseurl}}public/image-captioning.gif" width="400">

</div>

Minimal implementation of image captioning API hosted on [Heroku](https://image-captioning-69420.herokuapp.com/). It receives an image and responds with a caption regarding the image. This project's source code is hosted on [GitHub](https://github.com/reshalfahsi/image-captioning-api).


### SotoDemoBot

<div align="center">

<img src="{{site.baseurl}}public/sotodemobot.gif" width="400">

</div>

Simple Object detection Telegram bOt DEMO: predict the objects in the given image. Use ``/predict <URL>`` to predict the objects in the image of given url. This project's source code is hosted on [GitHub](https://github.com/reshalfahsi/object-detection-bot).


-----



## Natural Language Processing ##


### English-To-German Neural Machine Translation Using Transformer ###


<div align="center">

<img src="https://raw.githubusercontent.com/reshalfahsi/neural-machine-translation/master/assets/qualitative_result.png" width="1000">

The attention maps from each of the Transformer's heads. Almost every corresponding word pair (English-German) at each head pays attention mutually.
</div>

Neural Machine Translation (NMT) is a family model or an approach to solving machine translation problems through an artificial neural network, typically deep learning. In other words, the model is dispatched to translate a sequence of words from the source language to the target language. In this case, the source language would be English and the target would be German. To fabricate the model, the Transformer layers are leveraged. The NMT model is trained on the Multi30K dataset. The model is then assessed on a subset of the dataset, which is the Flickr 2016 test dataset. This project's source code is hosted on [GitHub](https://github.com/reshalfahsi/neural-machine-translation).


### Movie Review Sentiment Analysis Using CNN and MLP

<div align="center">

<img src="https://raw.githubusercontent.com/reshalfahsi/movie-review-sentiment-analysis/master/assets/neg_CNN0.png" width="600">

Visualization of the first layer of CNN on the negative review.
</div>

Audiences' reactions to the movie they have watched can be presented in a text format called reviews. These reviews can be polarized into two clusters: positive responses and negative responses. Using CNN and MLP, one can perform sentiment analysis on movie reviews to automatically recognize the viewer's tendency toward a particular movie. CNN is used for extracting the latent information within the text format. MLP leverages the extracted information and carries out the classification task. The CNN-MLP model is evaluated with Standford's IMBD Movie Review dataset. On the test set, the model achieves ``85.6%`` accuracy. This project's source code is hosted on [GitHub](https://github.com/reshalfahsi/movie-review-sentiment-analysis).


### Your Kind Friend Bot: Unreliable Chatbot

<div align="center">

<img src="{{site.baseurl}}public/ykfbot.gif" width="400">

</div>

Your typical kind friend who talk nonsense just to kill time. It can respond to text or image. Visit [here](https://t.me/yourkindfriendbot). This project's source code is hosted on [GitHub](https://github.com/reshalfahsi/ykfbot).


-----


## Audio Data ##


### AI Cover Song

<div align="center">
    <a href="https://youtu.be/wU6WnPl54HI">
        <img src="https://img.youtube.com/vi/wU6WnPl54HI/hqdefault.jpg" alt="Itsuki Nakano - Asmalibrasi (AI Cover)" width=400 />
    </a>

Itsuki Nakano - Asmalibrasi (AI Cover).
</div>

Cover your favorite song by your favorite singer. This project's source code is hosted on [GitHub](https://github.com/reshalfahsi/AI-Cover-Song).


### Music Genre Classification

<div align="center">

<img src="https://raw.githubusercontent.com/reshalfahsi/music-genre-classification/master/classification.png" width="600">

</div>

Classify input audio into a particular genre of music. First, the audio is preprocessed via [MFCC](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum). Next, using MLP, we obtain the probability distribution of 10 classes of music genres. Before applying MLP to the MFCC, the cepstral coefficients with the length of the number of sequence of time has to be averaged and subjected to [CMVN](https://en.wikipedia.org/wiki/Cepstral_mean_and_variance_normalization). This project's source code is hosted on [GitHub](https://github.com/reshalfahsi/music-genre-classification).


-----


## Graph Data ##

### Web Traffic Prediction via Temporal Graph Neural Network ###

<div align="center">

<img src="https://raw.githubusercontent.com/reshalfahsi/web-traffic-prediction/master/assets/qualitative_result.png" width="600">

The visitor prediction at one of the vital mathematics articles on Wikipedia.
</div>

Temporal Graph Neural Network or Temporal GNN is one of the variants of the GNN which handles the spatio-temporal data. The term "spatio-" refers to the nature of the graph that is closely related to the spatial relationship that exists in the image data (recall that an image is basically a graph), and the term "temporal" here indicates the data may be progressively changing in a sequence of time. In this project, the [Chebysev GCNN+LSTM](https://arxiv.org/pdf/1612.07659.pdf) module and the Wiki Maths dataset are leveraged, which are provided by [PyTorch Geometric Temporal](https://github.com/benedekrozemberczki/pytorch_geometric_temporal). The complete Temporal GNN model contains the Chebysev GCNN+LSTM module, followed by a fully connected layer. Here, the model is trained to predict the daily user visits to Wikipedia's vital mathematics articles (represented by nodes/vertices). The graph's characteristic in the dataset is non-heterogenous and static. The details of the dataset can be seen [here](https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/dataset.html#torch_geometric_temporal.dataset.wikimath.WikiMathsDatasetLoader). This project's source code is hosted on [GitHub](https://github.com/reshalfahsi/web-traffic-prediction).



### Graph Neural Network for Node Classification ###

<div align="center">

<img src="https://raw.githubusercontent.com/reshalfahsi/node-classification/master/assets/qualitative_result.gif" width="600">

The visualization of the embedding space of the nodes in the large graph in the course of the training process.
</div>

A graph neural network (GNN) is a type of neural network leveraged to handle graph data. One kind of graph data is a single graph that is large enough to contain a myriad of nodes. Later, we can attribute each node to well-qualified features and discriminate them accordingly. Then, by means of GNN, we can perform node classification on this large graph. The CORA dataset, the publicly available dataset for node classification on a large graph, is used in this tutorial. The graph feature extractor utilized in this tutorial consists of a sequence of ``ResGatedGraphConv``, ``SAGEConv``, and ``TransformerConv``, which are implemented by [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/index.html). The final classifier comprises MLP. This project's source code is hosted on [GitHub](https://github.com/reshalfahsi/node-classification).


-----


## Machine Learning ##


### PyTorch Depthwise Separable Convolution

<div align="center">

<img src="https://production-media.paperswithcode.com/methods/Screen_Shot_2020-05-31_at_10.30.20_PM.png" width="400">

</div>

PyTorch (unofficial) implementation of Depthwise Separable Convolution. This type of convolution is introduced by Chollet in [Xception: Deep Learning With Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357). This package provides ``SeparableConv1d``, ``SeparableConv2d``, ``SeparableConv3d``, ``LazySeparableConv1d``, ``LazySeparableConv2d``, and ``LazySeparableConv3d``. This package's source code is hosted on [GitHub](https://github.com/reshalfahsi/separableconv-torch).



### Neural Network

<div align="center">

<img src="https://4.bp.blogspot.com/-Anllqq6pDXw/VRUSesbvyAI/AAAAAAAAsrc/CIHz6vLsuTU/s800/computer_jinkou_chinou.png" width="300">

</div>

A naive implementation of a neural network. The code structure is heavily inspired by [PyTorch](https://github.com/pytorch/pytorch) and [TensorFlow](https://github.com/tensorflow/tensorflow). However, this package is used for educational purposes and is not intended to be adopted in production. This project's source code is hosted on [GitHub](https://github.com/reshalfahsi/neuralnetwork).


-----


## Robotics ##


### Rocket Trajectory Optimization Using REINFORCE Algorithm ###

<div align="center">

<img src="https://raw.githubusercontent.com/reshalfahsi/rocket-trajectory-optimization/master/assets/qualitative_rocket.gif" width="600">

The rocket successfully landed on the surface of the moon after hovering under the control of the learned policy from the REINFORCE algorithm.
</div>

In the context of machine learning, reinforcement learning (RL) is one of the learning paradigms involving interaction between agent and environment. Recently, RL has been extensively studied and implemented in the field of control theory. The classic example of a control theory problem is trajectory optimization, such as for spacecraft or rockets. Here, in the RL lingo, a rocket can be treated as an agent, and its environment would be outer space, e.g., the surface of the moon. The environment obeys the Markov Decision Process (MDP) property. The agent obtains a reward and observes a state based on the action that is given to the environment. The action taken by the agent is determined by the policy distribution that can be learned in the course of the training process. To learn the policy, one approach is to utilize the REINFORCE algorithm. This method is a policy gradient algorithm that maximizes the expected return (reward), incorporating Monte Carlo approximation. In practice, the gradient of the expected return will be our objective function to update our policy distribution. This project's source code is hosted on [GitHub](https://github.com/reshalfahsi/rocket-trajectory-optimization).


### Suction Arm Manipulator Robot

<div align="center">
    <a href="https://youtu.be/cmVsOR96NVk">
        <img src="https://github.com/reshalfahsi/arm-suction-sim/blob/master/img/simulation.gif?raw=true" width=400 />
    </a>
</div>

Simulate the Suction Arm Manipulator Robot to pick up daily objects inspired by the Amazon Robotics Challenge. This project's source code is hosted on [GitHub](https://github.com/reshalfahsi/arm-suction-sim).


-----


## Other Open Source Software ##

For a list of my open-source software, please take a look at my [GitHub](https://github.com/reshalfahsi).
