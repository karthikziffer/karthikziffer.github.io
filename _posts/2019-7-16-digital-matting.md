---
layout: post
title: "Digital Matting"
author: "Karthik"
categories: journal
tags: [documentation,sample]

---

<br>

Paper: [A Late Fusion CNN for Digital Matting](http://www.cad.zju.edu.cn/home/weiweixu/wwxu2019.files/3710.pdf)


<br>

This paper studies the structure of a deep convolutional neural network to predict the foreground alpha matte by taking a single RGB image as input. Our network is fully convolutional with two decoder branches for the foreground and background classification respectively. Then a fusion branch is used to integrate the two classification results which gives rise to alpha values as the soft segmentation result



Digital matting is to accurately extract the foreground object in an image for object-level image composition.

<blockquote class="imgur-embed-pub" lang="en" data-id="a/IeG5qUT"  ><a href="//imgur.com/a/IeG5qUT"></a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>





We assume that the observed image I is generated from three underlying images: the foreground image F, the background image B, and the alpha matte α, through the following model:
<br>

$$
I_p = α_pF_p + (1 − α_p)B_p

\\
\\

\text{where p represents a pixel across all images, and the value of α_p ∈ [0, 1]}
$$
 



A common approach to digital matting proceeds in three steps, namely, 

- Learn the foreground and background color models. 
- Compute the probabilities of each pixel belonging to the learned models.
- Obtain the alpha values 





We propose a fully convolutional network (FCN) for automatic image matting by taking a single RGB image as input. We achieve this goal by designing two decoder branches in the network for the
foreground and background classification, and then use a fusion branch to integrate the two classification results, which gives rise to the soft segmentation result



It is based on the observation that the classification branches can well predict the hard segmentation result, but have difficulties in predicting precise probabilities as the alpha values at the
pixel level. The two-decoder branch structure allows us to design a fusion branch to correct the residuals left in the classification branches. Moreover, our training loss encourages the two decoder branches to agree with each other at the hard segmentation part and leave the soft segmentation part to be corrected by the fusion branch.



<blockquote class="imgur-embed-pub" lang="en" data-id="a/8sIxGTQ"  ><a href="//imgur.com/a/8sIxGTQ"></a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>



The network is trained over three consecutive steps: segmentation network pretraining step, fusion network pretraining step and finally end-to-end joint training step whose training loss is imposed on the output alpha matte.

we try to predict the alpha values with the following fusion formula:

<br>
$$
α_p = β_p\vec{F_p} + (1 − β_p)(1 − \vec{B_p})
$$


where F_p and B_p represent the predicted foreground and background probability at pixel p.  βp is the blending weight predicted by the fusion network.

From the optimization perspective, the derivative of αp with respect to βp vanishes when
<br>
$$
\vec{B_p} + \vec{F_p} = 1
$$


First, the fusion network will focus on learning the transition region from the foreground to the background if the predictions of the foreground/background probability maps are accurate which is the bottleneck for solving the matting problem. 

Second, we can carefully design the loss function to encourage that the F_p + B_p  ≠ 1 within the
transition region, which can provide useful gradients to train the fusion network



## Segmentation Network

The segmentation network consists of one encoder and two decoders. The encoder extracts semantic features from the input image. The two decoders share the same encoded bottleneck and predict the foreground and background probability maps, respectively. Specifically, we use DenseNet-201 without the fully-connected layer head as our encoder. Each branch consists of five decoder blocks which correspond to the five encoder blocks, and the decoder block follows the design of feature pyramid network structure



## Training Loss



<blockquote class="imgur-embed-pub" lang="en" data-id="a/Dt8aZji"  ><a href="//imgur.com/a/Dt8aZji"></a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>





The training loss combines the **L1 loss**, the**L2 loss**, and the **cross-entropy loss**. In particular, we control
the behaviour of the training process of our network by setting different weights for different pixels according to the alpha matte.



We first measure the difference between the predicted probability values and the ground truth alpha values:

<br>
$$
L_d(\vec{F_p}) =   |\vec{F_p} - \alpha_p| , 0< \alpha_p < 1  \ \ \ \ or \ \ \ \ \  


                 (\vec{F_p} - \alpha_p)^2 , \alpha_p = 0,1
$$
<br>

The difference is chosen to be L1 inside transition regions so as to recover the details of the alpha matte there, while the L2 loss is used in the rest of the regions to penalize the possible segmentation error. We find this setting can well balance between the soft segmentation and the hard
segmentation

We also introduce the L1 loss on the gradients of the predicted alpha matte since it is beneficial to remove the over-blurred alpha matte after classification:

<br>
$$
L_g(\vec{F_p}) \  =  \ |\nabla_x (\vec{F_p}) - \nabla_x(\alpha_p)| \ + \  |\nabla_y(\vec{F_p}) - \nabla_y(\alpha_p) |
$$
<br>

The cross-entropy (CE) loss for the foreground classification branch at a pixel p is given by:

<br>
$$
CE(\vec{F_p}) \ = \ w_p . (-\hat{\alpha_p}.log(\vec{F_p}) - (1-\hat{\alpha_p}).log(1-\vec{F_p}))

\\

\text{The weight $w_p$ is set to 1 when $α_p$ = 1 or 0 and set to
0.5 when $α_p$ is in (0, 1)}
$$
<br>


The final loss function of the foreground classification branch with respect to an image is:

<br>
$$
L_p = \sum_p { CE(\vec{F_p}) + L_d (\vec{F_p}) + L_g(\vec{F_p})}
$$
<br>


For the background classification branch, its loss L_B can be simply computed by setting α_p = 1 − α_p. 

We also impose the L_F and L_B loss at each decoder block of two branches to further regulate the behaviour of the network.



## Fusion Network

The goal of the fusion network is to output β_p at pixels to fuse the foreground and background classification results.

It is a fully convolution network with five convolution layers and one sigmoid layer to compute the blending weights β_p. The input of the network consists of 

- Feature maps from the last block of the foreground and background decoders. 

- Feature from the convolution with the input RGB image. 

  We set the size of convolution kernel to 3 × 3 according to the exper-
  iments and found that the fusion network with this kernel
  size can better produce the details of the alpha matte.

  

### Training Loss


<br>


$$
L_u = \sum_p {w_p . |\beta_p.\vec{F_p} + (1-\beta_p)(1-\vec{B_p}) - \alpha_p|}

\\ 

\text{Specifically, the weights of pixels wp are set to 1 whenever 
0 < α_p < 1 , and 0 .1 otherwise.
}
$$

<br>


## Results

<blockquote class="imgur-embed-pub" lang="en" data-id="a/6kLMvi5"  ><a href="//imgur.com/a/6kLMvi5"></a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>







