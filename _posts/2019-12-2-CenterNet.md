---
layout: post
title: "CenterNet"
author: "Karthik"
categories: journal
tags: [documentation,sample]
image: andrew-NG-ffp.png
---




[CenterNet: Keypoint Triplets for Object Detection](https://arxiv.org/abs/1904.08189)

<br>



[![Screenshot-from-2019-12-22-11-37-51.png](https://i.postimg.cc/rFQ5XHvQ/Screenshot-from-2019-12-22-11-37-51.png)](https://postimg.cc/gnLx3gcZ) 

This paper presents an efficient solution which explores the visual patterns within each cropped region with minimal costs. We build our framework upon a representative one-stage keypoint-based detector named CornerNet. Our approach, named CenterNet, detects each object as a triplet, rather than a pair, of keypoints, which improves both precision and recall. Accordingly, we design two customized modules named cascade corner pooling and center pooling, which play the roles of enriching information collected by both top-left and bottom-right corners and providing more recognizable information at the central regions.

In the current era, one of the most popular flowcharts is anchor-based, which placed a set of rectangles with predefined sizes, and regressed them to the desired place with the help of ground-truth objects.

To overcome the drawbacks of anchor-based approaches, a keypoint-based object detection pipeline named CornerNet was proposed. It represented each object by a pair of corner keypoints, which bypassed the need of anchor boxes and achieved the state-of-the-art one-stage object detection accuracy. Nevertheless, the performance of CornerNet is still restricted by its relatively weak ability of referring to the global information of an object. That is to say, since each object is constructed by a pair of corners, the algorithm is sensitive to detect the boundary of objects, meanwhile not being aware of which pairs of keypoints should be grouped into objects.

CenterNet explores the central region that is close to the geometric center, with one extra keypoint. Our intuition is that, if a predicted bounding box has a high IoU(intersection over Union) with the ground-truth box, then the probability that the center keypoint in its central region is predicted as the same class is high, and vice versa. Thus, during inference, after a proposal is generated as a pair of corner keypoints, we determine if the proposal is indeed an object by checking if there is a center keypoint of the same class falling within its central region. 

Accordingly, for better detecting center keypoints and corners, we propose two strategies to enrich center and corner information, respectively.

### Center Pooling

Center pooling helps the center keypoints obtain more recognizable visual patterns within objects, which makes it easier to perceive the central part of a proposal. We achieve this by getting out the max summed response in both horizontal and vertical directions of the center keypoint on a feature map for predicting center keypoints. 

### Cascade corner pooling

Cascade corner pooling equips the original corner pooling module with the ability of perceiving internal information. We achieve this by getting out the max summed response in both boundary and internal directions of objects on a feature map for predicting corners. Empirically, we verify that such a two-directional pooling method is more stable, i.e., being more robust to feature-level noises, which contributes to the improvement of both precision and recall.

---

<br>

## Architecture



[![Screenshot-from-2019-12-22-11-38-05.png](https://i.postimg.cc/d1xgxnJ1/Screenshot-from-2019-12-22-11-38-05.png)](https://postimg.cc/N9RNy69q)

<br>

CornerNet produces two heatmaps: 

- A heatmap of top-left corners and a heatmap of bottom-right corners. The heatmaps represent the locations of keypoints of different categories and assigns a confidence score for each keypoint. Besides, it also predicts an **embedding** and a **group of offsets** for each corner. 
  - The embeddings are used to identify if two corners are from the same object. 
  - The offsets learn to remap the corners from the heatmaps to the input image. 

For generating object bounding boxes, top-k left-top corners and bottom-right corners are selected from the heatmaps according to their scores, respectively. 

Then, the distance of the embedding vectors of a pair of corners is calculated to determine if the paired corners belong to the same object. An object bounding box is generated if the distance is less than a threshold. The bounding box is assigned a confidence score, which equals to the average scores of the corner pair. 

A highly efficient alternative called CenterNet to explore the visual patterns within each bounding box. For detecting an object, our approach uses a triplet, rather than a pair, of keypoints. The drawback of CornerNet is solved by CenterNet.

Our approach only pays attention to the center information, the cost of our approach is minimal. Meanwhile, we further introduce the visual patterns within objects into the keypoint detection process by using **center pooling** and **cascade corner pooling**. 

We represent each object by a center keypoint and a pair of corners. Specifically, we embed a heatmap for the center keypoints on the basis of CornerNet and predict the offsets of the center keypoints. Then, we use the method proposed in CornerNet to generate top-k bounding boxes. However, to effectively filter out the incorrect bounding boxes, we leverage the detected center keypoints and resort to the following procedure: 

1.  Select top-k center keypoints according to their scores.
2. Use the corresponding offsets to remap these center keypoints to the input image. 
3. Define a central region for each bounding box and check if the central region contains center keypoints. Note that the class labels of the checked center keypoints should be same as that of the bounding box. 
4. If a center keypoint is detected in the central region, we will preserve the bounding box. The score of the bounding box will be replaced by the average scores of the three points, i.e., the top-left corner, the bottom-right corner and the center keypoint. If there are no center keypoints detected in its central region, the bounding box will be removed. 

`The size of the central region in the bounding box affects the detection results. For example, smaller central regions lead to a low recall rate for small bounding boxes, while larger central regions lead to a low precision for large bounding boxes.` 

Therefore, we propose a scale-aware central region to adaptively fit the size of bounding boxes. The scale-aware central region tends to generate a relatively large central region for a small bounding box, while a relatively small central region for a large bounding box. 

[![Screenshot-from-2019-12-22-11-42-55.png](https://i.postimg.cc/13Kmjx23/Screenshot-from-2019-12-22-11-42-55.png)](https://postimg.cc/NK5vKnQ3)

Suppose we want to determine if a bounding box **i** needs to be preserved. Let tl_x and tl_y denote the coordinates of the top- left corner of **i** and **br_x** and **br_y** denote the coordinates of the **bottom-right corner** of **i**. Define a central region **j**. Let **ctl_x** and **ctl_y** denote the coordinates of the **top-left corner** of **j** and **cbr_x** and **cbr_y** denote the coordinates of the **bottom- right corner** of **j**. Then **tl_x**, **tl_y**, **br_x**, **br_y**, **ctl_x**, **ctl_y**, **cbr_x** and **cbr_y** should satisfy the following relationship: 

[![Screenshot-from-2019-12-22-11-43-37.png](https://i.postimg.cc/zBZxbMDT/Screenshot-from-2019-12-22-11-43-37.png)](https://postimg.cc/yJjm2LZN)

where **n** is odd that determines the scale of the central region **j**

<br>

---

## Enriching Center and Corner Information

[![Screenshot-from-2019-12-22-11-38-22.png](https://i.postimg.cc/rp8mVyx4/Screenshot-from-2019-12-22-11-38-22.png)](https://postimg.cc/G45rzCM3)

### Center pooling. 

The geometric centers of objects do not necessarily convey very recognizable visual patterns. To address this issue, we propose center pooling to capture richer and more recognizable visual patterns.

The detailed process of center pooling is as follows: the backbone outputs a feature map, and to determine if a pixel in the feature map is a center keypoint, we need to find the maximum value in its both horizontal and vertical directions and add them together. By doing this, center pooling helps the better detection of center keypoints.

### Cascade corner pooling

Corners are often outside the objects, which lacks local appearance features.

It first looks along a boundary to find a boundary maximum value, then looks inside along the location of the boundary maximum value to find an internal maximum value, and finally, add the two maximum values together. By doing this, the corners obtain both the the boundary in- formation and the visual patterns of objects. 

[![Screenshot-from-2019-12-22-11-44-49.png](https://i.postimg.cc/VN6J9Lv8/Screenshot-from-2019-12-22-11-44-49.png)](https://postimg.cc/WdBpTV0W)

 

---

