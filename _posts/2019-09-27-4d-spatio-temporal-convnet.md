---
layout: post
title: "4D Spatio-Temporal Convnets"
author: "Karthik"
categories: journal
tags: [documentation,sample]
---



<br>

Paper: [4D Spatio-Temporal ConvNets](<https://arxiv.org/pdf/1904.08755.pdf>)

<br>



In many robotics and VR/AR applications, 3D-videos are readily-available sources of input (a continuous sequence of depth images, or LIDAR scans). However, those 3D-videos are processed frame-by-frame either through 2D convnets or 3D perception algorithms. In this work, we propose 4-dimensional convolutional neural networks for spatio-temporal perception that can directly process such 3D-videos using high-dimensional convolutions. For this, we adopt sparse tensors and propose the generalized sparse convolution that encompasses all discrete convolutions. 

<br>

The paper shows that on 3D-videos, 4D spatio-temporal convolutional neural networks are robust to noise. 

<br>

<blockquote class="imgur-embed-pub" lang="en" data-id="a/v11JV4z"  ><a href="//imgur.com/a/v11JV4z"></a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>



<br>

A 3D-video is a temporal sequence of 3D scans such as a video from a depth camera, a sequence of LIDAR scans, or a multiple MRI scans of the same object or a body part there are many technical challenges in using 3D- videos for high-level perception tasks. First, 3D data requires a heterogeneous representation and processing that either alienates users or makes it difficult to integrate into larger systems. Second, the performance of the 3D convolutional neural networks is worse or on-par with 2D convolutional neural networks. Third, the limited number of open-source libraries for fast large-scale 3D data is another factor. Out of these representations, we chose a sparse tensor due to its expressiveness and gen- eralizability for high-dimensional spaces. Also, it allows homogeneous data representation within traditional neural network libraries since most of them support sparse tensors.

<br>

Second, the sparse convolution closely resembles the standard convolution which is proven to be successful in 2D perception as well as 3D reconstruction, feature learning, and semantic segmentation. As the generalized sparse convolution is a direct high-dimensional extension of the standard 2D convolution, we can re-purpose all architectural innovations such as residual connections, batch normalization, and many others with little to no modification for high-dimensional problems. Third, the sparse convolution is efficient and fast. It only computes outputs for predefined coordinates and saves them into a compact sparse tensor. It saves both memory and computation especially for 3D scans or high-dimensional data where most of the space is empty. 

<br>

However, even with the efficient representation, merely scaling the 3D convolution to high-dimensional spaces results in significant computational overhead and memory con- sumption due to the curse of dimensionality. A 2D convolution with kernel size 5 requires 5^2 = 25 weights which increases exponentially to 5^3 = 125 in 3D, and 625 in 4D. This exponential increase, however, does not necessarily translate to better performance and slows down the network significantly. To overcome this challenge, we pro- pose custom kernels with non-(hyper)-cubic shapes. Finally, the 4D spatio-temporal predictions are not necessarily consistent throughout the space and time. To enforce consistency, the paper proposes the conditional random fields defined in a 7D trilateral space (space-time-color) with a stationary consistency function. Variational inference was used to convert the conditional random field to differentiable recurrent layers which can be implemented in as a 7D Minkowski network and train both the 4D and 7D networks end-to-end. 

<br>

The 4D spatio-temporal perception fundamentally requires 3D perception as a slice of 4D along the temporal dimension is a 3D scan. 

<br>

Sparse Tensor and Convolution in traditional speech, text, or image data, features are extracted densely. However, for 3-dimensional scans, such dense representation is inefficient since most of the space is empty. Instead, we can save non-empty space as its co-ordinate and the associated feature. This representation is an N-dimensional extension of a sparse matrix. we augment the 4D space with the chromatic space and create a 7D sparse tensor for trilateral filtering.

<br>

Generalized Sparse Convolution in this section, we generalize sparse convolutions proposed for generic input and output coordinates and for arbitrary kernel shapes. 

<br>

### Sparse Tensor Quantization 
The first step in the sparse convolutional neural network is the data processing to generate a sparse tensor, which converts an input into unique coordinates and associated features. If there are more than one different semantic labels within a voxel, we ignore this voxel during training by marking it with the IGNORE_LABEL. First, we convert all coordinates into hash keys and find the unique hashkey-label pairs to remove collision.

<br>

The next step in the pipeline is generating the output co-ordinates C_out given the input coordinates C_in. 

<br>

Next, to convolve inputs with a kernel, we need a mapping to identify which inputs affect which outputs. We call this mapping the kernel maps and define them as pairs of lists of input indices and output indices. Finally, given the input and output coordinates, the kernel map, and the kernel weights, we can compute the generalized sparse convolution by Minkowski Convolutional Neural Networks.

<br>

We introduce the 4-dimensional spatio-temporal convolutional neural network. We treat the time dimension as an extra spatial dimension and create a neural network with 4-dimensional convolutions. However, there are unique problems arising from such high-dimensional convolutions. 

- First, the computational cost and the number of parameters in a network increases exponentially as we increase the dimension. However, we experimentally show that these increases do not necessarily lead to better performance. 

- Second, the network does not have an incentive to make the prediction consistent throughout the space and time with conventional cross-entropy loss alone. 

  <br>

  To resolve the first problem, we make use of a special property of the generalized sparse convolution and propose non-conventional kernel shapes that save memory and computation with better generalization. Second, for spatio-temporal consistency, we propose a high-dimensional conditional random field (in 7D space-time-color space) that can enforce consistency and train both the base network and the conditional random field end-to-end. 

<br>

### Tesseract Kernel and Hybrid Kernel 

<blockquote class="imgur-embed-pub" lang="en" data-id="a/06elmbb"  ><a href="//imgur.com/a/06elmbb"></a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>

The surface area of 3D data increases linearly to time and quadratically to the spatial resolution. However, if we use a 4D hypercube, or a tesseract, for convolution kernels, the exponential increase in number of parameters most likely leads to over-parametrization, overfitting, as well as high computational-cost and memory consumption. 

<br>

For spatial dimensions, we use a cubic kernel to capture the spatial geometry accurately. And for the temporal dimension, we use the cross-shaped kernel to connect the same point in space across time. 

<br>

### Residual Minkowski Networks 
The generalized sparse convolution allows us to define strides and kernel shapes arbitrarily. Thus, we can create a high-dimensional network using the same generalized sparse convolutions homogeneously throughout the network, making the implementation easier and generic. 

<br>

### Trilateral Stationary-CRF 
The predictions from the MinkowskiNet for different time steps are not necessarily consistent throughout the temporal axis. To make such consistency more explicit and to improve predictions, we propose a conditional random field with a stationary kernel defined in a trilateral space. The trilateral space consists of 3D space, 1D time, and 3D chro- matic space; it is an extension of a bilaterl space in image processing. The color space allows points with different colors that are spatially adjacent (e.g. on a boundary) to be far apart in the color space. 

<br>

### Training and Evaluation 
We use Momentum SGD with the Poly scheduler to train networks from learning rate 1e-1 and apply data augmentation including random scaling, rotation around the gravity axis, spatial translation, spatial elastic distortion, and chro- matic translation and jitter. For evaluation, we use the standard mean Intersection over Union (mIoU) and mean Accuracy (mAcc).

<br>

### Conclusion 
In this paper, we propose a generalized sparse convolution and an auto-differentiation library for sparse tensors. Using these, we create a 4D convolutional neural network for spatio-temporal perception. Experimentally, we show that 3D convolutional neural networks can outperform 2D networks and 4D perception can be more robust to noise. 

<br>