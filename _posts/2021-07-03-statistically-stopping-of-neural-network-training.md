---
layout: post
title: "Statistically stopping of neural network training"
author: "Karthik"
categories: journal
tags: [documentation,sample]


---



Paper: [Statistically Significant Stopping of Neural Network Training](https://arxiv.org/abs/2103.01205)

Github: [Code Repository](https://github.com/justinkterry/ASWS)



<br>



![](https://media.giphy.com/media/1yTgp9l8CbatRiEQ77/giphy.gif)

<br>

Much learning of neural network does not take place once the optimal values are found, the condition does not impact the final accuracy of the model. 

According to the runtime perspective, this is of great significance when numerous neural networks are trained simultaneously. 

This paper introduces a statistical significance test to determine if a neural network has stopped learning. 

Additionally, this method can be used as a new learning rate scheduler. 

<br>



---



Currently, the optimal place to stop the neural network's training is when the test data error is minimum. Existing solution such as early stopping looks for

-  No change in test accuracy over a given number of epochs
- Absolute change in accuracy or loss
- A decrease in highest test accuracy observed over a given number of epochs
- An average change in test accuracy over a given number of epochs

<br>

In the context of Auto ML, along with above conditions, two of the most popular conditions are used.

- Median stopping rule: Where a run is stopped if its performance falls below the median of other trials at similar points in time. 
- Hyper Band: Where runs are placed into brackets and low performing runs in each bracket are stopped.



This paper introduces a statistical significance test to determine if a neural network has stopped learning, by only looking at the testing set accuracy curve. The test used in this paper is an extension of <mark>Shapiro Wilk test</mark>, and named as <mark>Augmented Shapiro Wilk Stopping (ASWS)</mark>. 

This method stops in 77% or less steps than all popular conditions. Other methods stop too early at an expense of 2-4% final accuracy even with tuned hyperparameter.

<br>



---



#### Some Background

![Alt Text](https://media.giphy.com/media/l2JhORT5IFnj6ioko/giphy.gif)

<br>

<mark>Shapiro Wilk Test</mark> determines the probability that a sample of data points was drawn from a normal distribution. It is the most powerful normality test. 

<mark>Single Sample T-test</mark> determines the probability that a sample of data points was drawn from a distribution with a mean other than a specified one.

<mark>Clipped Exponential Smoothing</mark> is a method for smoothing time series data. 

<br>



---



##### Augmented Shapiro Wilk Stopping

<mark>While training, accuracy on the test dataset will be increasing, with a high degree of noise from random sampling of the data, and numeric errors amongst other sources. When the variations in the test accuracy curve become purely noise and their mean is zero, then you can be fairly confident that learning has stopped</mark>. 

Per the central limit theorem, when these variations are random they will also be normally distributed. The Shapiro Wilk test can tell if the recent accuracy values are normally distributed, and the simple sample t-test will tell you if they have zero mean.

<mark>The problem with this is the nature of the noise during training. If the noise of an error curve is too extreme, then any changes will become washed out</mark>. Furthermore, the noise seen is very dependent on the neural networks loss landscape, meaning that the variations are not Independent and Identically Distributed (I.I.D) random variables on small time scales. These factors make any statistical analysis very challenging. 

There are three mitigations to the problem of noise together, good results can still be achieved. 

- Smooth error curves. Sensible amounts of smoothing can't meaningfully change the macro trends we're looking at here, and removing the overall level of noise will make it much easier to find.
- Only look at the testing set accuracy curve. Overcoming the much greater noise and non I.I.D characteristics of loss curves remains a future work.
- Check for both 0 mean and normal distribution at once. Both should be true when a neural network is done training, and testing for both dramatically decreases false positive rates. 

<br>

---



Results

The only stopping methods able to consistently achieve higher test accuracy, when compared to ASWS method are the performance stopping method and the average difference stopping method. The difference in test accuracy is 0.5%. 

This paper shows that the ASWS learning rate scheduler can achieve comparable performance to schedulers which are commonly used in fewer iterations. All of these advances are of potentially great use towards amore environmentally sustainable machine learning, faster prototyping, and less far computational expense during large AutoML training endeavours. 



<br>



![](https://media.giphy.com/media/3o7buiyYnf8OhsVp9m/giphy.gif)



<br>









