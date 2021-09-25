---
layout: post
title: "Serving Tensorflow to 300 million predictions per second"
author: "Karthik"
categories: journal
tags: [documentation,sample]



---

This [paper](https://arxiv.org/pdf/2109.09541.pdf) presents the process of transitioning machine learning models to the Tensorflow framework at a large scale in an online advertising ecosystem. 



<br>

![Alt Text](https://media.giphy.com/media/3FjwOam0L3bcVERzO9/giphy.gif?cid=ecf05e47csxd64410s2nr9gg0l46kv3cj21yirs5e0dwmafl&rid=giphy.gif&ct=g)

<br>

I will outline the important practices to follow while training and serving a machine learning model

<br>

The process of scaling machine learning models implemented in the Tensorflow machine learning framework to over 300 million predictions per second at Zemanta, an Outbrain company. 

Zemanta is a demand side platform(DSP) in the real time bidding(RTB) ecosystem, which is a fast growing part of online advertising. In RTBs, several DSPs(bidders) compete for advertising space online by bidding for it in real time while a web page is still loading. 

The advertising space is sold on a per-ad impression basis, which enables selling virtual ad space at market value. <mark>Through the use of machine learning, RTB also enables advertisers to maximize their KPIs such as click through rate. Estimating the CTR of ads is therefore one of the central problems in RTB since it allows advertisers to only bid and pay for measurable user responses, such as clicks on ads</mark>. Having a good click prediction model is thus of significant importance. 

The RTB field comes with a few intrinsic properties: Large amounts of data, a low latency requirement. 

The distribution of the data also changes rapidly, meaning that models need to be updated with new data frequently in order to stay competitive. 

<br>

---

<br>

### Challenges:

Since the bidder was a monolith, which is beneficial in three areas: 

- ease of deployment, latency and engineering costs. 
- A series of services would have higher or less predictable latency. 

<mark>We used the core TF framework inside the bidder application</mark>. An alternative is Tensorflow Serving, which is a premade service for serving TF models with additional features such as batching capabilities. 

<mark>We do not utilize GPUs for inference in production</mark>. At our scale, outfitting each machine with one or more top class GPUs would be prohibitively expensive and on the other hand, having only a small cluster of GPU machines would force us to transition to a service based architecture. 

Our use case is also not a good fit for GPU workloads due to our models using sparse weights. 

<br>

---

<br>

### Implementations:

TF offers a massive ecosystem and plenty of libraries with state of the art algorithm implementations. It is very easy to pick a feature-rich-off-the-shelf implementations, however, we found that these are mostly unoptimized. 

We then decided to implement the algorithms ourselves, but even starting was not trivial. TF has APIs of varying levels of abstraction, from the most easy to use, but often inefficient to the most low level operations. <mark>We chose Keras as it is a thin wrapper around low level TF operations and maintains a high level of performance while being easy to understand</mark>. Since TF is very feature rich and resource heavy library, we also had to consider how much of our machine learning pipeline we would implement in it. We opted to set aside feature transformation and interaction for now and only implement learning algorithms - they are the smallest part that can be replaced yet offer the highest potential for improvement. 

<mark>Because the Golang TF wrapper supports only predictions, we had to implement the training loop in Python</mark>. The script is connected to our Golang data pipeline through its standard input as a subprocess. <mark>Data is sent to it in a highly efficient binary format, requiring no parsing - this was a 25% improvement in speed over a CSV format</mark>. 

The data is then read in a background thread to prevent the model from being idle while waiting for data. With this, we managed to retain a high throughput through the entire training pipeline, having only the model as a potential bottleneck. 

<mark>Efficient inputs and outputs proved to be key for low latency prediction as well, where we significantly decreased the time spent on expensive serialization and copying of input data by joining all input features into a single tensor</mark>. 

<br>

---

<br>

### Serving:

<mark>We found that by using the Golang TF wrapper out of the box, the DeepFM models incurred a much higher CPU usage due to the compute intensive neural networks. Despite bringing a significant lift in business metrics, scaling this approach to 100% of our traffic would have required a significant hardware investment</mark>

To combat this, we saw a need to make these computations less expensive. Reducing the model's neural network size was to be avoided if possible as it would also reduce the model's predictive performance. 

<mark>By diving deeply into TF, we realized that the computation is far more efficient, if we increase the number of examples in a compute batch. This low linear growth is due to TF code being highly vectorized. </mark>

TF also has some overhead for each compute cell, which is then amortized over larger batches. 

Given this, <mark>we figured that in order to decrease the number of compiute calls, we needed to join many requests into a single computation. </mark>

<mark>We built an autobatching system contained entirely within a running bidder instance,avoiding network calls. Since each instance receives thousands of bid requests per second, we can reliably join the computations from many requests, creating bigger batches. </mark>

We did this by having a few batcher threads which receive data from incoming requests, create batches and initialize computation once the batch has been filled. 

<mark>The computation is always initialized at least every few milliseconds to prevent timeouts since it is possible that the batch isn't filled in this time window. This implementation is highly optimized and is able to decrease the number of compute calls by a factor of 5, halving the CPU usage of TF compute. </mark>

In rare cases that a batcher thread does not get CPU time, those requests will time out. However, this happens on fewer than 0.01% of requests. We observed a slight increase in the average latency - by around 5 millisecond on average, which can be higher in peak traffic. 

<mark>We put SLAs and appropriate monitoring into place to ensure stable latencies. As we did not increase the percentage of timeouts substantially, this was highly beneficial and is still the core of our TF serving mechanisms. </mark>

<br>

---

<br>

### Optimization:

The models we implemented in TF were initially much slower than the custom built FMs. To find potential speedups, <mark>we heavily utilized the inbuilt TF profiler to find the operations which take the longest to execute. </mark>

We were able to create many possible improvements with this insights, the most common being various redundant reshape or transform operations. 

<mark>One of the most interesting findings was discovering that the Adam optimizer was much slower than Adagrad (around 50%), despite the difference in the number of operations being small. </mark>

<mark>The profiler showed that gradient updates on our sparse weights require a large amount of computational time. This is due to the model's weights being sparse (the features are largely categorical and thus very sparse) and the optimizer not taking this fact into account. </mark> Since replacing Adam with Adagrad meant a significant deterioration of the deep model's performance, we looked for other solutions. 

<mark>Switching to the lazy Adam Optimizer proved very beneficial as it handles sparse weights very efficiently. We found that it sped up overall training over 40%, bringing it up to par with Adagrad in this regard. </mark>

<mark>In RTB, the data distribution changes rapidly, presenting a need for continuous model training.</mark> This is why we continuously update our models with a job and deploy the trained model onto the fleet of bidder machines. Because we run many models in production at the same time, the memory and storage requirements are significant. <mark>Since we use adaptive optimizers such as Adam, this also requires storing weight's moments and variances - instead of one, three values are stored for each parameter, increasing the saved model size threefold.</mark> However, these values are not actually needed for prediction, only for training. We utilized this to construct an optimization routine that strips the model of these values, reducing the amount of data that is pulled to our bidder machines by 66% and decreasing the memory usage and costs.

<br>

---

<br>

### Conclusion:

We described the process of transitioning machine learning models to the Tensorflow framework and serving them at a large scale. 

<mark>The key challenges we faced in our use case were related to compute resources, prediction latency, and training throughput. </mark>

<mark>By implementing autobatching in serving, we halved TF's CPU usage and retained acceptable latencies. </mark>

Along with thoroughly understanding the models, we also put effort into putting together an efficient training pipeline: 

- using a binary data format instead of CSV, utilizing the TF profiler to remove bottlenecks in the models and using the correct optimizer have all brought significant speedups. 
- We have also implemented many other smaller and more specific optimization such as stripping the optimizer weights to reduce the saved model size. 

Overall , <mark>using TF has brought significant lifts in business metrics and vastly increased the speed of research. </mark>

To make the best use of it, we are continuing to optimize our pipelines and serving stack. 

<br>

![Alt Text](https://media.giphy.com/media/DAzIIpUmSgFXbp1hFV/giphy.gif?cid=ecf05e47645f250q7hhxdbujxbstor14bix3z7knaxvv1laz&rid=giphy.gif&ct=g)



