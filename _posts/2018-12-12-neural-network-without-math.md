---
layout: post
title: "Neural Network without Math"
author: "Karthik"
categories: journal
tags: [documentation,sample]
---





Let's cover the fundamentals.

### What is Convolution?

Convolution uses kernels/filter to extract information from images. Different kernels can extract different representations from images. For different tasks such as sharpening , edge detection , blurring different kernels can be employed.  Kernel is matrix which slides upon the image to extract local neighbouring representations and transform the captured representations through convolution at the output.

### A good Analogy

It is impossible to read the image and understand its contents by closing our eyes. But now, we have an image surface consisting of it's content edges. These edges are elevated above the image surface and are distinguishable from the surface. Now, by sliding our hand through the surface,  we can find out the edges representation and try to decode the contents present on the surface without seeing the image. Convolution also performs similar operation using kernels.

### Why we use convolution?

- A raw image contains noise, to normalize the noise we can use convolution. By selecting appropriate kernel matrix, respective noise normalized output image is obtained.
- Convolution extract underlying image features using different application specific kernels.







## Forward Propagation

---





In neural network, Edges are extracted from input image pixels. From edges we further extract Patterns. From Patterns we predict Output class labels.

The input image is a grid of Pixel values. Pixels come together to form representations in an image. These representations maybe wheels in cars, Ears of Dogs etc. These representations are absorbed by the neural network. The more number of variational images the neural network comes across, more variational representation is grasped by the network.

Through convolution, the pixels representation are captured by kernel parameters (kernel matrix elements). The pixel values from the input images are transformed into pixel sequence. We multiply weights with every pixels from this pixel sequence.   Then by taking the pixel values(input activations) and weights, we compute the weighted sum.

The weighted sum computed will be any number.

Now, To boundary the weighted sum within a range we use activation function such as relu, sigmoid etc. The range value will between different activation functions.

Activation output is the measure of "How positive the weighted sum is?"

In case of sigmoid , If we want the activation to light up beyond the threshold value instead of zero. Then we introduce an additional parameter called Bias along with the weighted sum.  This Bias is an inactive measure. An Activation gets lighted up only by crossing this Bias.

In neural networks, the kernel values are random initialized at beginning of forward propagation.  Then through multiple epochs during training , these kernel values(weights) get updated with the notion of reducing prediction loss. The steps taken to reduce the loss is called learning rate.



## Back Propagation

---







When the kernels are randomly initialized in the first forward pass, most probably the predicted output will be inaccurate. Now the difference between the predicted and actual input is calculated. This difference is called the loss. Our objective by training a neural network is to reduce this loss, So that the actual label will be predicted by the network. Over time, our network predicts the label with confidence value. We train the model to be more confident for predicting the actual label.

Since the network is interconnected from output layer to input layer through hidden layers, It is unchallenging to update  activations in the hidden layers proportional to the loss incurred in the output layer.  For instance, the output layer is connected to immediate predecessor (hidden) layer. This loss difference is reflected on this immediate predecessor's activations. We update the weights which influences the predecessor activations. This pattern follows till the input layer.  This updates are carried out until we train the model.

This is the complete overview of neural network without any mathematics.