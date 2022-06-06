---
layout: post
title: "Machine Learning and Deep Learning comparison"
author: "Karthik"
categories: journal
tags: [documentation,sample]




---



[A Comparison of Traditional Machine Learning and Deep Learning in Image Recognition](https://iopscience.iop.org/article/10.1088/1742-6596/1314/1/012148)



<br>

This paper compares the accuracy by using machine learning and the Deep convolution neural network. The paper used a computationally simpler DCNN architecture. 

<br>

#### The ML technique: 

[Bag of words model in computer vision](https://en.wikipedia.org/wiki/Bag-of-words_model_in_computer_vision)

```
In computer vision, the bag-of-words model (BoW model) sometimes called bag-of-visual-words model [1][2] can be applied to image classification or retrieval, by treating image features as words. In document classification, a bag of words is a sparse vector of occurrence counts of words; that is, a sparse histogram over the vocabulary. In computer vision, a bag of visual words is a vector of occurrence counts of a vocabulary of local image features.

```

<br>



[Scale invariant feature transform](https://en.wikipedia.org/wiki/Scale-invariant_feature_transform)

```
> The SIFT features are local and based on the appearance of the object at particular interest points, and are invariant to image scale and rotation. 
> They are also robust to changes in illumination, noise, and minor changes in viewpoint. 
> In addition to these properties, they are highly distinctive, relatively easy to extract and allow for correct object identification with low probability of mismatch.
```



<br>

The ML technique comprises of extracting key features using SIFT algorithm and creates a global representation using Bag of visual words model. Use the representation vector for each image as input features and classify the output. Perform classification using algorithms such as K-nearest neighbors, Support vector machine.   

<br>

#### The Deep Learning Technique:

DL experiment are carried out with and without Pre-trained model. 

<br>

#### Dataset:

There was class imbalance in BelgiumTS - Belgium Traffic Sign Dataset, but this did not affect the quality of the data. 

<br> 

##### BOVW model

The BOVW consists of a dictionary, constructed by a clustering algorithm which aims to locate differences between an image and a general representation of a dataset. 

The operating principle behind the BOVW model supports that to encode all the local features of an image, a universal representation of an image must be created. This model compares the examined image with the generated representation of each class and generates an output based on the differences of their content. 

<br>

##### Encoding

Another step of great importance is to determine the properties of the classifier. Specifically, this is achieved via encoding the content of the images based on a dictionary of universal characteristics. In order to perform this, a histogram is produced that provides information regarding the frequency of the visual words of the dictionary in an image.  

Moreover, upon producing a histogram for each word - using a vector of features - images are compared with a dictionary, and words correspond to the shortest distance. This results in finding the greatest similarity between the dataset. 

Finally, we notice that normalization is applied to the calculation of the occurrence frequency as we wished to ensure that the generated histograms did not depend on the number of visual words. 

<br>

##### KNN classifier

The KNN algorithm is a non-parametric classifier which accepts the histograms of the previous stage and compares them with the image dataset focusing on calculating and monitoring differences in the measured distances. Then, each image is classified to a unique cluster which shows the greatest degree of similarity with its K nearest neighbors.

The classifier depends greatly on the distance metric used to predict and categorize each set of results into k-groups. The distance measure selected highly depends on the dataset examined and should be chosen after a trail and error approach.   

Some distances are Manhattan distance, Euclidean distance. In this study Minkowski distance was used. 

```
The Minkowski distance or Minkowski metric is a metric in a normed vector space which can be considered as a generalization of both the Euclidean distance and the Manhattan distance.
```

[Minkowski distance](https://en.wikipedia.org/wiki/Minkowski_distance)

<br>

##### SVM classifier

The SVM classifier uses an algorithm to select the surface that lies equidistant from the 2 nearest vector space. This is achieved via classifying dataset into different classes and calculating the margin between each class, thus creating vectors that support this margin region. 

The support vectors of the data points that are closer to the hyperplane influence the position and orientation of the hyperplane.

<br>

#### Deep Learning Algorithm:

The most common use of neural networks in the computer vision domain are CNNs implementing hierarchical feature learning. From the experiments, data augmentation and normalization methods are used to avoid overfitting. 

VGG16 architecture:

The VGG16 architecture followed by our custom classifier. Fully connected layers with batch normalization and dropout for regularization and mish activation function. 

```
Mish activation provides a smooth gradient which improves the rate of convergence
```

[Mish Activation](https://arxiv.org/pdf/1908.08681.pdf)

<mark>Here all the weights of each layer of VGG16 architecture frozen except for the last 4 layers. Since the first convolution layers learn low level features which are similar in most of the images. The deeper layers are trained to learn high level features and are problem specific. </mark>

The classifier stage after the feature extraction stage consists of 4 fully connected layers. The aim of fully connected layers are to achieve optimal performance as it is proposed for larger and deeper networks, as due to the multiple stacked convolution layers, more complex features of the input volume can be extracted. 

<mark>As the network progresses deeper, the number of applied filters is augmented. Initially we start with 16 filters and gradually increase them to 32 and 64. This increase assists in producing high quality features as it combines low level features that occur while training the network. </mark>

In DL, it is common to double the number of channels after each pooling layer as the knowledge of each layer becomes more spatial. 

<mark>The pooling layers divides the spatial dimensions by 2, this reduces the risk of rapid increases in parameters, memory usage and computing load. </mark>

Mish activation was the optimal choice after experiments. Mish activations generated the smaller loss rates and a smoother loss landscape representation as its behavior suggests that it avoids saturation due to capping. 

<mark>A smooth activation function allow optimal information propagation deeper into the neural network, thus achieving higher accuracy and generalization results. </mark>

<br>

#### Conclusion:

This paper proves that Deep Learning model provides higher accuracy than Machine Learning model. 

Training from scratch and pretrained model are capable of achieving similar results with existing CNNs but authors suggest that for medium to small size datasets. 