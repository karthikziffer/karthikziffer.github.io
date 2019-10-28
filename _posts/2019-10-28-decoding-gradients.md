---
layout: post
title: "Decoding gradients"
author: "Karthik"
categories: journal
tags: [documentation,sample]

---


<br>




In neural networks, the convergence is attained by finding the optimal network error. The error is defined as a function called the Loss function, which must be minimized through an iterative process of weight updates w.r.t the gradients.

![Screenshot-from-2019-10-28-11-47-23.png](https://i.postimg.cc/rwsrKvyv/Screenshot-from-2019-10-28-11-47-23.png)

Gradients are defined as the rate of change of Loss function w.r.t the multivariate weights. 

An indepth understanding of optimization can be understood by **Introduction to deep learning course by CMU**


### The problem of Optimization

![Screenshot-from-2019-10-28-10-46-54.png](https://i.postimg.cc/nzNMN5YD/Screenshot-from-2019-10-28-10-46-54.png)

The minimum value of Loss function facilitates the network with convergence. But in general, the loss function is a non-convex function which makes the process of finding the lowest point complicated.  

By employing gradient of the multivariate function we can navigate to the minimum point. There are possibilities that the local minimum can be misunderstood as global minimum during the optimization process. 
  
### Gradient 

![Screenshot-from-2019-10-28-10-53-04.png](https://i.postimg.cc/F15dkqY4/Screenshot-from-2019-10-28-10-53-04.png)

![Screenshot-from-2019-10-28-10-57-13.png](https://i.postimg.cc/43brDh6b/Screenshot-from-2019-10-28-10-57-13.png)

The rate of change of Loss function w.r.t all the weights in the network provides a comprehensive dependency of network weights in minimizing the loss. 

### Properties of Gradient

![Screenshot-from-2019-10-28-11-15-02.png](https://i.postimg.cc/zBLTL4Jd/Screenshot-from-2019-10-28-11-15-02.png)


![Screenshot-from-2019-10-28-11-24-05.png](https://i.postimg.cc/cL9mrRRL/Screenshot-from-2019-10-28-11-24-05.png)

![Screenshot-from-2019-10-28-11-22-59.png](https://i.postimg.cc/zfvxjtzh/Screenshot-from-2019-10-28-11-22-59.png)

When both the vector are aligned i.e when the value of **θ** = 0 and **Cos(θ)** = 1.  Hence the both vectors must be aligned to move in the fastest changing diection. The direction is obtained by the gradient sign.

The sign plays the major role in navigating towards the minimum/maximum loss point in the non-convex loss function. 

![Screenshot-from-2019-10-28-11-35-27.png](https://i.postimg.cc/Jn4HJvRw/Screenshot-from-2019-10-28-11-35-27.png)
 

![Screenshot-from-2019-10-28-11-36-52.png](https://i.postimg.cc/s281QXHW/Screenshot-from-2019-10-28-11-36-52.png)


![Screenshot-from-2019-10-28-11-38-16.png](https://i.postimg.cc/zXJDfKDM/Screenshot-from-2019-10-28-11-38-16.png)


![Screenshot-from-2019-10-28-11-39-16.png](https://i.postimg.cc/52DM8C1f/Screenshot-from-2019-10-28-11-39-16.png)

![Screenshot-from-2019-10-28-11-42-22.png](https://i.postimg.cc/fyyHH8jp/Screenshot-from-2019-10-28-11-42-22.png)


![Screenshot-from-2019-10-28-11-43-17.png](https://i.postimg.cc/dt7HrS0T/Screenshot-from-2019-10-28-11-43-17.png)

![Screenshot-from-2019-10-28-11-44-06.png](https://i.postimg.cc/3JD1d4LX/Screenshot-from-2019-10-28-11-44-06.png)


![Screenshot-from-2019-10-28-11-46-33.png](https://i.postimg.cc/qBKQ76D8/Screenshot-from-2019-10-28-11-46-33.png)

