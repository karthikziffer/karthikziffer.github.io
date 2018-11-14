---
layout: post
title: "Summary - Gradient Descent Finds Glabal Minimum"
author: "Karthik"
categories: journal
tags: [documentation,sample]
image:
---




[Gradient Descent Finds Global Minima of Deep Neural Networks](https://arxiv.org/pdf/1811.03804.pdf)

- Simon S. Du
- Jason D. Lee
- Haochuan Li
- Liwei Wang
- Xiyu Zhai


---



### Abstract
> Gradient descent finds a global minimum in training deep neural networks despite the objective function being non-convex. The current paper proves gradient descent achieves zero training loss in polynomial time for a deep over-parameterized neural network with residual connections (ResNet). 

The graph of loss function across weights/parameters in the network being a non-convex (i.e) with multiple local minimum and only single global minimum, Gradient Descent tries to find the global minimum during the training process.  

> Our analysis relies on the particular structure of the Gram matrix induced by the neural network architecture. This structure allows us to show the Gram matrix is stable throughout the training process and this stability implies the global optimality of the gradient descent algorithm. 

#### Introduction

1. One of the mysteries in deep learning is random initialized first order methods like gradient
descent achieve zero training loss, even if the labels are arbitrary [Zhang et al., 2016].
<br>
2. Overparameterization is widely believed to be the main reason for this phenomenon as only if the neural network has a sufficiently large capacity, it is possible for this neural network to fit all the training data. 
<br>
3. The second mysterious phenomenon in training deep neural networks is “deeper networks are harder to train.” To solve this problem, He et al. [2016] proposed the deep residual network (ResNet) architecture which enables randomly initialized first order method to train neural networks with an order of magnitude more layers. Theoretically, Hardt and Ma [2016] showed that residual links in linear networks prevent gradient vanishing in a large neighborhood of zero, but for neural networks with non-linear activations, the advantages of using residual connections are not well understood.


---

##### In this paper, we demystify these two mysterious phenomena.
We consider the setting where there are **n** data points, and the neural network has **H** layers with width **m**.

- We first consider a fully-connected feedforward network. 
 We show if m = Ω(poly(n)2O(H)), then randomly initialized gradient descent converges to zero training loss at a linear rate.

-  Next, we consider the ResNet architecture. We show as long as m = Ω (poly(n, H)), then randomly initialized gradient descent converges to zero training loss at a linear rate. Comparing with the first result, the dependence on the number of layers improves exponentially for ResNet. This theory demonstrates the advantage of using residual connections.

-  Lastly, we apply the same technique to analyze convolutional ResNet. 
We show if m = poly(n, p, H)  where p is the number of patches, then randomly initialized gradient descent achieves zero training loss.

---
#### Related Works

[Saddle Point from Wikipedia](https://en.wikipedia.org/wiki/Saddle_point) - In mathematics, a saddle point or minimax point is a point on the surface of the graph of a function where the slopes (derivatives) in orthogonal directions are all zero (a critical point), but which is not a local extremum of the function. 

> Recently, many works try to study the optimization problem in deep learning. Since optimizing a neural network is a non-convex problem, one approach is first to develop a general theory for a class of non-convex problems which satisfy desired geometric properties and then identify that the neural network optimization problem belongs to this class. One promising candidate class is
the set of functions that satisfy all local minima are global and there exists a negative curvature for every saddle point. For this function class, researchers have shown gradient descent [Jin et al., 2017, Ge et al., 2015, Lee et al., 2016, Du et al., 2017a] can find a global minimum
[Saddle Point](https://en.wikipedia.org/wiki/Saddle_point)

The above approach says about finding the global minimum by considering saddle point with negative curvature.

> Many previous works thus try to study the optimization landscape of neural networks with different activation functions [Safran and Shamir, 2018, 2016, Zhou and Liang, 2017, Freeman and Bruna, 2016, Hardt and Ma, 2016, Nguyen and Hein, 2017, Kawaguchi, 2016, Venturi et al., 2018, Soudry and Carmon, 2016, Du and Lee, 2018, Soltanolkotabi et al., 2018, Haeffele and Vidal, 2015]. However, even for a deep linear network, there exists a saddle point that does not have a negative curvature [Kawaguchi, 2016], so it is unclear whether this approach can be used to obtain the global convergence guarantee of first-order methods.
[Negative curvature](http://stanwagon.com/wagon/misc/htmllinks/invisiblehandshake_3.html)

The Second approch describes a scenario where there is possibilty of saddle point without a negative curvature. In this situation the first approach will fail.


This Paper follows the below mentioned approach.
> Another way to attack this problem is to study the dynamics of a specific algorithm for a specific
neural network architecture. Our paper also belongs to this category.

> Many previous works put assumptions on the input distribution and assume the label is generated according to a planted neural network. Based on these assumptions, one can obtain global convergence of gradient descent for some shallow neural networks [Tian, 2017, Soltanolkotabi, 2017, Brutzkus and Globerson, 2017, Du et al., 2018a, Li and Yuan, 2017, Du et al., 2017b].

>Some local convergence results have also been proved [Zhong et al., 2017a,b, Zhang et al., 2018].

> In comparison, our paper does not try to recover the underlying neural network. Instead, we focus the empirical loss minimization problem and rigorously prove that randomly initialized gradient descent can achieve zero training loss.


> The most related papers are Li and Liang [2018], Du et al. [2018b] who observed that when training a two-layer full connected neural network, most of the patterns do not change over iterations, which we also use to show the stability of the Gram matrix. They used this observation to obtain the convergence rate of gradient descent on a two-layer over-parameterized neural network for the cross-entropy and least-squares loss.


> More recently, Allen-Zhu et al. [2018] generalizes ideas from Li and Liang [2018] to derive convergence rates of training recurrent neural networks.

> Our work extends these previous results in several ways: 
	>>	1. we consider deep networks.
	>>	2. we generalize to ResNet architectures.
	>>	3. we generaliez to convolutional networks.


> Chizat and Bach [2018], Wei et al. [2018], Mei et al. [2018] used optimal transport theory to analyze gradient descent on over-parameterized models. However, their results are limited to twolayer neural networks and may require an exponential amount of over-parametrization.


> Daniely [2017] developed the connection between deep neural networks with kernel methods and showed stochastic gradient descent can learn a function that is competitive with the best function in the conjugate kernel space of the network. 

> Andoni et al. [2014] showed that gradient descent can learn networks that are competitive with polynomial classifiers. However, these results do not imply gradient descent can find a global minimum for the empirical loss minimization problem.


#### Conclusion
> In In this paper, we show that gradient descent on deep overparametrized networks can obtain zero training loss. The key technique is to show that the Gram matrix is increasingly stable under overparametrization, and so every step of gradient descent decreases the loss at a geometric rate.

#### List of some directions for future research.

1. The current paper focuses on the train loss, but does not address the test loss. It would be an important problem to show that gradient descent can also find solutions of low test loss. In particular, existing work only demonstrate that gradient descent works under the same situations as kernel methods and random feature methods [Daniely, 2017, Li and Liang, 2018.

2. The width of the layers m is polynomial in all the parameters for the ResNet architecture, but still very large. Realistic networks have number of parameters, not width, a large constant multiple of n. We consider improving the analysis to cover commonly utilized networks an important open problem.

3. The current analysis is for gradient descent, instead of stochastic gradient descent. We believe the analysis can be extended to stochastic gradient, while maintaining the linear convergence rate.


---
---
### Note
All the Mathematical derivations and calculations are skipped in this article. Please refer the paper for More Info.


