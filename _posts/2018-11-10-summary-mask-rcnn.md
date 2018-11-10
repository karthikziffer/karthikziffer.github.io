---
layout: post
title: "Summary - Mask RCNN"
author: "Karthik"
categories: journal
tags: [documentation,sample]
image:
---


[MASK R-CNN](https://arxiv.org/pdf/1703.06870.pdf)
Facebook AI Research (FAIR)
- Kaiming He 
- Georgia Gkioxari 
- Piotr Dollar 
- Ross Girshick ´ 
---


What does Mask R-CNN do 
> We present a conceptually simple, flexible, and general framework for object instance segmentation. Our approach efficiently detects objects in an image while simultaneously generating a high-quality segmentation mask for each instance.

What is the additional feature of Mask R-CNN from Faster R-CNN
 > The method, called Mask R-CNN, extends Faster R-CNN by adding a branch for predicting an object mask in parallel with the existing branch for bounding box recognition.

How are Mask R-CNN results
> Moreover, Mask R-CNN is easy to generalise to other tasks, e.g., allowing us to estimate human poses in the same framework. We show top results in all three tracks of the COCO suite of challenges, including instance segmentation, boundingbox object detection, and person keypoint detection. Without bells and whistles, Mask R-CNN outperforms all existing, single-model entries on every task, including the COCO 2016 challenge winners.

Where can I find Mask R-CNN code
> [Code has been made available here](https://github.com/facebookresearch/Detectron)

---
**Lets understand the terminology used in this paper**
1. Object Detection - Detection of object via bounding boxes.
2. Semantic Segmentation - Denotes per-pixel classification without differentiating instances.
3. Instance Segmentation - Combination of both Object Detection and Semantic Segmentation with instance object masking.


--- 
How Mask R-CNN extends from Faster R-CNN

![](https://pli.io/2ZHLy8.png)

> Our method, called Mask R-CNN, extends Faster R-CNN [36] by adding a branch for predicting segmentation masks on each Region of Interest (RoI), in parallel with the existing branch for classification and bounding box regression (Figure 1). The mask branch is a small FCN applied to each RoI, predicting a segmentation mask in a pixel-to pixel manner. 

> Mask R-CNN is simple to implement and train given the Faster R-CNN framework, which facilitates a wide range of flexible architecture designs. Additionally, the mask branch only adds a small computational overhead, enabling a fast system and rapid experimentation.

What Faster R-CNN was lacking

> In principle Mask R-CNN is an intuitive extension of Faster R-CNN, yet constructing the mask branch properly is critical for good results. Most importantly, Faster RCNN was not designed for pixel-to-pixel alignment between network inputs and outputs. This is most evident in how RoIPool [18, 12], the de facto core operation for attending to instances, performs coarse spatial quantization for feature extraction.

How ROI Pool works
The ROI Pool is a neural network layer which takes two inputs. One is the feature maps and bounding box co-ordinates. The bounding box co-ordinates are used to draw bounding boxes on the feature maps and the respective region of interest is obtained. Next the ROI is scaled to output dimension. Irrespective of ROI dimension, it will be scaled to output dimension using max-pooling. This performs spatial quantization which speeds up the detection process but in turn loosing pixel to pixel alignment information.
[Detailed Explanation and visualization of ROI pooling](https://deepsense.ai/region-of-interest-pooling-explained/)


How Mask R-CNN overcomes this 
>To fix the misalignment, we propose a simple, quantization-free layer, called RoIAlign, that faithfully preserves exact spatial locations. Despite being a seemingly minor change, RoIAlign has a large impact: it improves mask accuracy by relative 10% to 50%, showing bigger gains under stricter localization metrics.


Additional Mask R-CNN Features
> Second, we found it essential to decouple mask and class prediction: we predict a binary mask for each class independently, without competition among classes, and rely on the network’s ROI classification branch to predict the category.

>> The **binary mask** defines a region of interest (ROI) of the original image. **Mask** pixel values of 0 indicate the image pixel is part of the background. **Mask** pixel values of 1 indicate the image pixel is not part of the background.  [Source](https://www.mathworks.com/help/images/create-binary-mask-from-grayscale-image.html)


In Depth Understanding

> Mask R-CNN is conceptually simple: Faster R-CNN has two outputs for each candidate object, a class label and a bounding-box offset; to this we add a third branch that outputs the object mask. Mask R-CNN is thus a natural and intuitive idea. But the additional mask output is distinct from the class and box outputs, requiring extraction of much finer spatial layout of an object. Next, we introduce the key elements of Mask R-CNN, including pixel-to-pixel alignment, which is the main missing piece of Fast/Faster R-CNN.

Faster R-CNN
> Faster R-CNN: We begin by briefly reviewing the Faster R-CNN detector [36]. Faster R-CNN consists of two stages. The first stage, called a Region Proposal Network (RPN), proposes candidate object bounding boxes. The second stage, which is in essence Fast R-CNN [12], extracts features using RoIPool from each candidate box and performs classification and bounding-box regression. The features used by both stages can be shared for faster inference. We refer readers to [21] for latest, comprehensive comparisons between Faster R-CNN and other frameworks.


Mask R-CNN
>Mask R-CNN: Mask R-CNN adopts the same two-stage procedure, with an identical first stage (which is RPN). In the second stage, in parallel to predicting the class and box offset, Mask R-CNN also outputs a binary mask for each RoI. This is in contrast to most recent systems, where classification depends on mask predictions (e.g. [33, 10, 26]). Our approach follows the spirit of Fast R-CNN [12] that applies bounding-box classification and regression in parallel (which turned out to largely simplify the multi-stage pipeline of original R-CNN [13]).


>Formally, during training, we define a multi-task loss on each sampled RoI as 
$$L = L_{cls} + L_{box} + L_{mask}$$ The classification loss Lcls and bounding-box loss Lbox are identical as those defined in [12].

> The mask branch has a $$Km^2$$  dimensional output for each RoI, which encodes K binary masks of resolution $$m × m$$, one for each of the K classes.

>To this we apply a per-pixel sigmoid, and define Lmask as the average binary cross-entropy loss. For an ROI associated with ground-truth class k, Lmask is only defined on the k-th mask (other mask outputs do not contribute to the loss).

> Our definition of Lmask allows the network to generate masks for every class without competition among classes;

> we rely on the dedicated classification branch to predict the class label used to select the output mask. This decouples mask and class prediction.

> This is different from common practice when applying FCNs [30] to semantic segmentation, which typically uses a per-pixel softmax and a multinomial cross-entropy loss. In that case, masks across classes compete;

> In our case, with a per-pixel sigmoid and a binary loss, they do not. We show by experiments that this formulation is key for good instance segmentation results.

#### Mask Representation
>A mask encodes an input object’s spatial layout.

> Thus, unlike class labels or box offsets that are inevitably collapsed into short output vectors by fully-connected (fc) layers, extracting the spatial structure of masks can be addressed naturally by the pixel-to-pixel correspondence provided by convolutions.

>Specifically, we predict an m × m mask from each RoI using an FCN [30]. This allows each layer in the mask branch to maintain the explicit m × m object spatial layout without collapsing it into a vector representation that lacks spatial dimensions.

>Unlike previous methods that resort to fc layers for mask prediction [33, 34, 10], our fully convolutional representation requires fewer parameters, and is more accurate as demonstrated by experiments.

> This pixel-to-pixel behavior requires our RoI features, which themselves are small feature maps, to be well aligned to faithfully preserve the explicit per-pixel spatial correspondence. This motivated us to develop the following RoIAlign layer that plays a key role in mask prediction.


#### ROI Align

>RoIPool [12] is a standard operation for extracting a small feature map (e.g., 7×7) from each RoI. RoIPool first quantizes a floating-number RoI to the discrete granularity of the feature map, this quantized RoI is then subdivided into spatial bins which are themselves quantized, and finally feature values covered by each bin are aggregated (usually by max pooling). Quantization is performed, e.g., on a continuous coordinate x by computing [x/16], where 16 is a feature map stride and [·] is rounding; likewise, quantization is performed when dividing into bins (e.g., 7×7). These quantizations introduce misalignments between the RoI and the extracted features. While this may not impact classification, which is robust to small translations, it has a large negative effect on predicting pixel-accurate masks.

>To address this, we propose an RoIAlign layer that removes the harsh quantization of RoIPool, properly aligning the extracted features with the input. Our proposed change is simple: we avoid any quantization of the RoI boundaries or bins (i.e., we use x/16 instead of [x/16]).

> We use bilinear interpolation to compute the exact values of the input features at four regularly sampled locations in each RoI bin, and aggregate the result (using max or average). We note that the results are not sensitive to the exact sampling locations, or how many points are sampled, as long as no quantization is performed.

> RoIAlign leads to large improvements. We also compare to the RoIWarp operation proposed in [10]. Unlike RoIAlign, RoIWarp overlooked the alignment issue and was implemented in [10] as quantizing RoI just like RoIPool. So even though RoIWarp also adopts bilinear resampling motivated by [22], it performs on par with RoIPool , demonstrating the crucial role of alignment.

#### Network Architecture

>We instantiate Mask R-CNN with multiple architectures. For clarity, we differentiate between: (i) the convolutional backbone architecture used for feature extraction over an entire image, and (ii) the network head for bounding-box recognition (classification and regression) and mask prediction that is applied separately to each ROI.

> We denote the backbone architecture using the nomenclature network-depth-features. We evaluate ResNet and ResNeXt  networks of depth 50 or 101 layers. The original implementation of Faster R-CNN with ResNets  extracted features from the final convolutional layer of the 4-th stage, which we call C4. This backbone with ResNet-50, for example, is denoted by ResNet-50-C4. 

> We also explore another more effective backbone recently proposed by Lin et al, called a **Feature Pyramid Network** (FPN). FPN uses a top-down architecture with lateral connections to build an in-network feature pyramid from a single-scale input. Faster R-CNN with an FPN backbone extracts ROI features from different levels of the feature pyramid according to their scale, but otherwise the rest of the approach is similar to vanilla ResNet. Using a ResNet-FPN backbone for feature extraction with Mask RCNN gives excellent gains in both accuracy and speed. For further details on FPN, we refer readers to [27].

> For the network head we closely follow architectures presented in previous work to which we add a fully convolutional mask prediction branch.

> Specifically, we extend the Faster R-CNN box heads from the ResNet [19] and FPN [27] papers. Details are shown in Figure 4. The head on the ResNet-C4 backbone includes the 5-th stage of ResNet (namely, the 9-layer ‘res5’ [19]), which is computeintensive. For FPN, the backbone already includes res5 and thus allows for a more efficient head that uses fewer filters.

![HEAD ARCHITECTURE](https://pli.io/2ZFLFY.png)


#### Implementation details

>We set hyper-parameters following existing Fast/Faster R-CNN work. Although these decisions were made for object detection in original papers, we found our instance segmentation system is robust to them.


##### Training

>As in Fast R-CNN, an RoI is considered positive if it has IoU with a ground-truth box of at least 0.5 and negative otherwise.

> The mask loss Lmask is defined only on positive RoIs. The mask target is the intersection between an RoI and its associated ground-truth mask.

> We adopt image-centric training. Images are resized such that their scale (shorter edge) is 800 pixels.

> Each mini-batch has 2 images per GPU and each image has N sampled RoIs, with a ratio of 1:3 of positive to negatives.

> N is 64 for the C4 backbone (as in [12, 36]) and 512 for FPN (as in [27]). We train on 8 GPUs (so effective minibatch size is 16) for 160k iterations, with a learning rate of 0.02 which is decreased by 10 at the 120k iteration. We use a weight decay of 0.0001 and momentum of 0.9. With ResNeXt, we train with 1 image per GPU and the same number of iterations, with a starting learning rate of 0.01.

##### Inference

> The mask branch can predict K masks per RoI, but we only use the k-th mask, where k is the predicted class by the classification branch. The m×m floating-number mask output is then resized to the RoI size, and binarized at a threshold of 0.5.

>Note that since we only compute masks on the top 100 detection boxes, Mask R-CNN adds a small overhead to its Faster R-CNN counterpart (e.g., ∼20% on typical models).


#### Result
![Experiment Outcomes](https://pli.io/2ZFp9H.png)



---
### Resources:
1. [Ferature Pyramid Network](https://arxiv.org/pdf/1612.03144.pdf)

