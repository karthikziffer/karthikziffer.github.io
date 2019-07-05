---
layout: post
title: "Class Activation Map"
author: "Karthik"
categories: journal
tags: [documentation,sample]

---



<br>





[Learning Deep Features for Discriminative Localization](http://cnnlocalization.csail.mit.edu/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf)

<br>



---



We know that Convolution Neural Networks are good at classification tasks. This paper decodes how previous layer activation contribute for localization tasks even though the network is being trained on classification tasks.  Using Global Average pooling layer, the localization ability is studied.  

Advantage of Global Average Pooling extends beyond regularization in the network.   A CNN trained for classification task can also localize the descriminative regions. 

---

<br>

Two works are mostly related to this paper:

#### Weakly supervised object localization.

- Global Average pooling provides the advantage of capturing entire object boundary for localization over Global max pooling which captures single point within the object boundary.  

- The Class activation maps is used to refer the weighted activation maps generated for each image.  

#### Visualizing CNNs.

- Visualizing the discriminative regions provides transparent view about the CNN. This assists in concluding which region was responsible for classification output, additionally what representation the network has captured. 

---

<br>



### Generating Class Activation Map

Generation of Class activation maps using Global Average Pooling in CNN is described in this Paper. 

By performing global average pooling on the convolution feature maps and use those as features for a fully connected layer that produce the desired output. 

We can identify the importance of the image regions by projecting back the weights of the output layer on to the convolution feature maps, a technique we call class activation mapping.  

Global Average pooling outputs the spatial average of the feature map of each unit at the last convolution layer. A weighted sum of these values is used to generate the final output. 

Similarly, we compute a weighted sum of feature maps of the last convolution layer to obtain our class activation maps. 

<br>



<blockquote class="imgur-embed-pub" lang="en" data-id="a/dcFFNCB" data-context="false" ><a href="//imgur.com/a/dcFFNCB"></a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>



- For a given image, Let

$$
f_k(x,y)
$$
represent the activation of unit **k** in the last convolution layer at spatial location (x,y)

- Then, for unit **k**, the result of performing global average pooling, 

$$
F_k = \sum_{x,y} f_k(x,y)
$$

- 

$$
S_c = \sum_{k} w_k^c.F_k
\\
w_k^c \ weight \ corresponding \ to \ class \ c \ for \ unit \ k
$$

- Output of the softmax for class c , 

$$
P_c = e^{S_c} \div \sum_{c} e^{S_c}
$$



- We define , Class Activation Map

$$
M_c(x,y) = \sum_k w_k^c. f_k(x,y)
\\
M_c \ is \ the \ class \ activation \ map \
$$

- Softmax class output dependency on Class Activation Map is given by

$$
S_c = \sum_{x,y} M_c(x,y)
$$



---

<br>









Reference:

- [Blog post 1](https://alexisbcook.github.io/2017/global-average-pooling-layers-for-object-localization/)
- [Blog post 2](https://harrisonjansma.com/GAP)