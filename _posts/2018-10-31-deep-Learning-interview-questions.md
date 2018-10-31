---
layout: post
title: "Deep Learning interview questions."
author: "Karthik"
categories: journal
tags: [documentation,sample]
---

 

### Difference between Deep Networks vs Shallow Networks ?

*Deep Networks* - The number of hidden layers are more in deep networks with large number of parameters. Since there are higher number of parameters, higher the degree of non-linearity in the network. Hence this increases the capability to extract high level features. 

*Shallow Networks* - The number of hidden layers are less, hence less number of parameters. Therefore high level features cannot be extracted. Since the number of parameters are less, this model is less computationally expensive.

---

### What is cost function ? 

The measure of difference in accuracy between the actual training sample and expected output. This provides the information about the gap in accuracy that our network needs to cover in order to predict the actual output. When the network's predicted output and training sample's actual output are same, then the cost function is zero. This is an ideal scenario.

---

### What is Gradient descent ?

Gradient descent is an optimization algorithm which is used to learn the value of parameters that minimizes the cost function. It is an iterative algorithm which moves in the direction of steepest descent to reach the global loss minima. The negative sign, denotes the downward direction carried out by the optimizer to reduce the network's loss function by finding the global minima.

---

### What is Back Propagation ?

Back propagation is used for multi layer neural network. In this method, we move the error from an end of the network to all weights inside the network and thus allowing efficient computation of the gradient. The main intention of back propagation is to tune the previous layer's activation function. Since directly varing the activation function is not possible. But by updating the parameters in the respective hidden layer which constitutes the activation function, We can tune the hidden layer activations. 

---

### List the end to end steps carried out in neural network. 

1. Forward propagation of training data in order to generate output.

2. Then using target value and output value, error derivative(Cost function) can be computed with respect to output activation. 

3. Then we back propagate for computing derivative of error with respect to output activation on previous and continue this for all the hidden layers.

4. Using previously calculated derivatives for output and all hidden layers we calculate error derivatives with respect to weights.

5. And then we update the parameters(weights).

     

   ---

### Resources: 

   1. [3Blue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
   2. [Springboard Blog](https://www.springboard.com/blog/machine-learning-interview-questions/)

