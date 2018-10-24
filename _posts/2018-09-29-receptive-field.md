---
layout: post
title: "Intro to Receptive field from a Research Paper"
author: "Karthik"
categories: journal
tags: [documentation,sample]
---
 

<!-- # Intro to Receptive field from a Research Paper -->



##### Understanding the Effective Receptive Field in Deep Convolutional Neural Networks



- The receptive field size is a crucial issue in many visual tasks , as the output must respond to large enough areas in the image to capture information about large objects. The effective receptive field show that it both has Gaussian distribution and only occupies a fraction of the full theoretical receptive field.

- This [paper](http://www.cs.toronto.edu/~wenjie/papers/nips16/top.pdf) analyze effective receptive field in several architecture designs, and effect of nonlinear activations , dropout , sub-sampling and skip connections.

---

- One of the basic concepts in deep CNNs is the receptive field or field of view , of a unit in a certain layer in the network. Unlike in fully connected networks, where the value of each unit depends on the entire input to the network, a unit in convolutional networks only depends on a region of the input.This region in the input is the receptive field for that unit.
- The concept of receptive field is important for understanding and diagnoising how deep CNNs work. Since anywhere in an input image outside the receptive field of a unit does not affect the value of that unit, it is necessary to carefully control the receptive field, to ensure that it covers the entire relevant image region.
- The receptive field size of a unit can be increased in a number of ways. One option is to stack more layers to make the network deeper, which increases the receptive field size linearly by theory, as each extra layer **increases the receptive field size by the kernel size**.
- Sub-sampling on the other hand increases the receptive field size multiplicatively.
- Intuitively it is easy to see that pixels at the center of a receptive field have a much larger impact on an output. In the forward pass, central pixels can propagate information to the output through many different paths, while the pixels in the outer area of the receptive field have very few paths to propagate its impact. In the backward pass, gradients from an output unit are propagated across all the paths, and **therefore the central pixels have a much larger magnitude for the gradient from that output.**
- There's an intriguing findings, in particular that the effective area in the receptive field, which we call the effective receptive field, only occupies a fraction of the theoretical receptive field, since Gaussian distribution generally decay quickly from the center.



