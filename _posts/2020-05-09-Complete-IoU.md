---
layout: post
title: "Complete IoU"
author: "Karthik"
categories: journal
tags: [documentation,sample]

---










The loss functions are the major driving force in training a good model. For Object detection and Instance segmentation tasks, the most widely used loss function is Intersection over Union (IOU). In [Enhancing Geometric Factors for Object Detection and Instance Segmentation Loss function.](https://arxiv.org/pdf/2005.03572.pdf) psper, a new loss function called as Complete Intersection over Union id proposed by considering three geometric factors.

<br>

### Intersection over Union

Intersection over Union is the ratio of Area of Overlap over Area of Union. 


$$
IOU = \frac{|A \cap B|}{|A \cup B}
$$


![Screenshot-2020-05-09-at-5-52-05-PM.png](https://i.postimg.cc/d31dBsGc/Screenshot-2020-05-09-at-5-52-05-PM.png)



<br>



#### How does IoU loss help in detection or segmentation model training? 



It is the ratio of predicted bounding box overlapping over the ground truth area box. There are two extreme scenarios here. The positive scenario, where both the box overlap 100%, then the IoU ratio will be 1. On the other hand, the negative scenario, when the predicted box is far away from the ground truth box without any overlap, then the IoU value will be 0. By taking (1 - IoU) the loss will become maximum. The performance of the model must be improved by reducing this loss. 

IoU has scale invariance property. This means, the width, height and location of the two bounding boxes are taken into consideration. The normalized IoU measure focuses on the area of the shapes, no matter their size. 


The problem to distinguish between the same level of overlap, but different scales will give different values. State of the art object detection networks deal with this problem by introducing ideas such as anchor boxes and non-linear representations.  


<br>


### Generalized Intersection over Union


$$
GIoU = \frac{|A \cap B|}{|A \cup B} - \frac{C\setminus( A \cup B)}{|C|} \ = IoU - \frac{C\setminus( A \cup B)}{|C|}
$$


Here, A and B are the prediction and ground truth bounding boxes. C is the smallest convex hull that encloses both A and B. C is the smallest box covering A and B. 

The penality term in GIoU loss, will move the predicted box towards the target box in non-overlapping cases.



---

<br>



In this paper, a new CIoU loss is introduced by considering the geometric factors in bounding box regression. This loss as improved the average precision and average recall without the sacrifice of inference efficiency. 



<br>



### Comparision of GIoU and CIoU

GIoU loss tries to maximize overlap area of two boxes and still performs limited due to only considering overlap areas. 

GIoU loss tends to increase the size of the predicted box, while the predicted box moves towards the target box very slowly. Consequently , GIoU loss emprically needs more iterations to converge, especially for bounding box at horizontal and vertical orientations. Thus increasing the training time.

<br>


![Screenshot-2020-05-09-at-6-55-30-PM.png](https://i.postimg.cc/xd7MpByb/Screenshot-2020-05-09-at-6-55-30-PM.png)





---

<br>

CIoU depend on three geometric factors for modelling regression relationships. 


$$
CIoU = S (A, B) + D(A, B) + V(A, B)
\\
$$

```
S - Overlap Area
D - Normalized central point distance
V - Aspect Ratio
```



These three geometric factors are incorporated into CIoU loss for better distinguishing difficult regression cases. 

For detection tasks, generally (Ln-norm) Mean squared Error loss and Smooth loss is been widely used for object detection, pedestrian detection, pose estimation and instance segmentation. However, recent works suggest that Ln-norm based loss functions are not consistent with the evaluation metric, instead propose Intersection over Union. 



### Overlap Area

The previous IoU and GIoU loss has proven overlap area calculation. Hence the overlap area employs the same IoU loss. 
$$
S = 1 - IoU
$$


### Normalized central point distance



![Screenshot-2020-05-09-at-7-30-51-PM.png](https://i.postimg.cc/63FmLHm9/Screenshot-2020-05-09-at-7-30-51-PM.png)



<br>

<br>

In the above overlapping scenarios, the distance might vary across different training observations. To solve this, the distance is normalized.

Both Normalized central point distance and Aspect ratio must be invariant to regression scale, Hence the normalized central point distance to measure the distance of two boxes is employed. 

<br>

![Screenshot-2020-05-09-at-7-16-56-PM.png](https://i.postimg.cc/QML0HBwz/Screenshot-2020-05-09-at-7-16-56-PM.png)


$$
D = \frac{(Euclidean \ distance \ ( A, B))^2}{c^2}
$$


### Aspect ratio


$$
V = \alpha \ (\frac{4}{\pi^2}(arctan(\frac{w^B}{h^B}) - arctan(\frac{w}{h}))^2)
\\ 

w^B \ width \ of \ ground \ truth \ box
\\
h^B \ height of ground truth box
$$



<br>


$$
\alpha = 0, \\if \ IoU < 0.5
$$



<br>


$$
\alpha = \frac{V}{(1-IoU) + V} \\ if \ IoU \ge 0.5
$$


<br>



alpha is a trade-off parameter. When the IoU is less than 0.5, then the two boxes are not well matched, the consistency of aspect ratio is less important.

CIoU loss can provide moving direction for bounding boxes when non-overlapping with target box. 



### Loss Visualization



![Screenshot-2020-05-09-at-7-52-12-PM.png](https://i.postimg.cc/hGW8rkSZ/Screenshot-2020-05-09-at-7-52-12-PM.png)



<br>





This is a new loss for improving the training of detection and segmentation tasks. CIoU converges faster with less iterations than IoU and GIoU due to the consideration of three geometric factors. 



---



### Resources

1. https://giou.stanford.edu/

   



 

  