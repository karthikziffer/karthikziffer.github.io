---
layout: post
title: "DiffusionDet Inference"
author: "Karthik"
categories: journal
tags: [documentation,sample]






---



<br>

## DiffusionDet

<br>

In this blog post I will provide the procedures that I followed to inference the DiffusionDet Model for object detection. This is based on the github repo https://github.com/ShoufaChen/DiffusionDet

<br>

I created a conda environment 

```
conda create --name diffusionDetection python=3.7
```

Detectron repo needs python >= 3.7



<br>



Activate the conda environment

```
conda activate diffusionDetection
```

<br>

Clone the DiffusionDet repo

```
git clone https://github.com/ShoufaChen/DiffusionDet.git
```

<br>

DiffusionDet repo does not have the requirements.txt file, I had to individually install the pip packages and set up the environment. 

```
pip install numpy
pip install opencv-python
pip install tqdm
```

<br>

I want to try the inference on a CPU only machine, hence I installed PyTorch without CUDA support. PyTorch is a dependency for the Detectron2

```
conda install pytorch cpuonly -c pytorch
```

<br>

Install Detectron2 

```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

You can find alternative installation options from https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md#installation

<br>

Manually download the [pretrained model](https://github.com/ShoufaChen/DiffusionDet/releases/download/v0.1/diffdet_coco_res50_300boxes.pth) and place it in the same folder as demo.py

<br>

Edit the config file Base-DiffusionDet inside the configs folder to add the MODEL.DEVICE as "cpu", since the inference is perfomed on CPU machine without GPU.  I only added the line  <mark> DEVICE: "cpu" </mark>.

<br>

```
MODEL:
  META_ARCHITECTURE: "DiffusionDet"
  DEVICE: "cpu"
  WEIGHTS: "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
  PIXEL_MEAN: [123.675, 116.280, 103.530]
  PIXEL_STD: [58.395, 57.120, 57.375]
  BACKBONE:
    NAME: "build_resnet_fpn_backbone"
  RESNETS:
    OUT_FEATURES: ["res2", "res3", "res4", "res5"]
  FPN:
    IN_FEATURES: ["res2", "res3", "res4", "res5"]
  ROI_HEADS:
    IN_FEATURES: ["p2", "p3", "p4", "p5"]
  ROI_BOX_HEAD:
    POOLER_TYPE: "ROIAlignV2"
    POOLER_RESOLUTION: 7
    POOLER_SAMPLING_RATIO: 2
SOLVER:
  IMS_PER_BATCH: 16
  BASE_LR: 0.000025
  STEPS: (210000, 250000)
  MAX_ITER: 270000
  WARMUP_FACTOR: 0.01
  WARMUP_ITERS: 1000
  WEIGHT_DECAY: 0.0001
  OPTIMIZER: "ADAMW"
  BACKBONE_MULTIPLIER: 1.0  # keep same with BASE_LR.
  CLIP_GRADIENTS:
    ENABLED: True
    CLIP_TYPE: "full_model"
    CLIP_VALUE: 1.0
    NORM_TYPE: 2.0
SEED: 40244023
INPUT:
  MIN_SIZE_TRAIN: (480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800)
  CROP:
    ENABLED: False
    TYPE: "absolute_range"
    SIZE: (384, 600)
  FORMAT: "RGB"
TEST:
  EVAL_PERIOD: 7330
DATALOADER:
  FILTER_EMPTY_ANNOTATIONS: False
  NUM_WORKERS: 4
VERSION: 2

```

<br>

Execution code

```
python demo.py --config-file configs/diffdet.coco.res50.yaml --input image.jpg --opts MODEL.WEIGHTS .\diffdet_coco_res50_300boxes.pth
```



<br>

Output

```
(diffusionDetection2) PS C:\Users\karth\OneDrive\Documents\Other_Resources\ProfileProject\DiffusionDet> python demo.py --config-file configs/diffdet.coco.res50.yaml --input image.jpg --opts MODEL.WEIGHTS .\diffdet_coco_res50_300boxes.pth

[11/25 00:30:07 detectron2]: Arguments: Namespace(confidence_threshold=0.5, config_file='configs/diffdet.coco.res50.yaml', input=['image.jpg'], opts=['MODEL.WEIGHTS', '.\\diffdet_coco_res50_300boxes.pth'], output=None, video_input=None, webcam=False)
cpu

[11/25 00:30:08 fvcore.common.checkpoint]: [Checkpointer] Loading from .\diffdet_coco_res50_300boxes.pth ...

[11/25 00:30:25 d2.checkpoint.c2_model_loading]: Following weights matched with model:
| Names in Model                                   | Names in Checkpoint                                                                                  | Shapes                                          |
|:-------------------------------------------------|:-----------------------------------------------------------------------------------------------------|:------------------------------------------------|
| alphas_cumprod                                   | alphas_cumprod                                                                                       | (1000,)                                         |
| alphas_cumprod_prev                              | alphas_cumprod_prev                                                                                  | (1000,)                                         |
| backbone.bottom_up.res2.0.conv1.*                | backbone.bottom_up.res2.0.conv1.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (64,) (64,) (64,) (64,) (64,64,1,1)             |
| backbone.bottom_up.res2.0.conv2.*                | backbone.bottom_up.res2.0.conv2.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (64,) (64,) (64,) (64,) (64,64,3,3)             |
| backbone.bottom_up.res2.0.conv3.*                | backbone.bottom_up.res2.0.conv3.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (256,) (256,) (256,) (256,) (256,64,1,1)        |
| backbone.bottom_up.res2.0.shortcut.*             | backbone.bottom_up.res2.0.shortcut.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight} | (256,) (256,) (256,) (256,) (256,64,1,1)        |
| backbone.bottom_up.res2.1.conv1.*                | backbone.bottom_up.res2.1.conv1.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (64,) (64,) (64,) (64,) (64,256,1,1)            |
| backbone.bottom_up.res2.1.conv2.*                | backbone.bottom_up.res2.1.conv2.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (64,) (64,) (64,) (64,) (64,64,3,3)             |
| backbone.bottom_up.res2.1.conv3.*                | backbone.bottom_up.res2.1.conv3.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (256,) (256,) (256,) (256,) (256,64,1,1)        |
| backbone.bottom_up.res2.2.conv1.*                | backbone.bottom_up.res2.2.conv1.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (64,) (64,) (64,) (64,) (64,256,1,1)            |
| backbone.bottom_up.res2.2.conv2.*                | backbone.bottom_up.res2.2.conv2.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (64,) (64,) (64,) (64,) (64,64,3,3)             |
| backbone.bottom_up.res2.2.conv3.*                | backbone.bottom_up.res2.2.conv3.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (256,) (256,) (256,) (256,) (256,64,1,1)        |
| backbone.bottom_up.res3.0.conv1.*                | backbone.bottom_up.res3.0.conv1.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (128,) (128,) (128,) (128,) (128,256,1,1)       |
| backbone.bottom_up.res3.0.conv2.*                | backbone.bottom_up.res3.0.conv2.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (128,) (128,) (128,) (128,) (128,128,3,3)       |
| backbone.bottom_up.res3.0.conv3.*                | backbone.bottom_up.res3.0.conv3.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (512,) (512,) (512,) (512,) (512,128,1,1)       |
| backbone.bottom_up.res3.0.shortcut.*             | backbone.bottom_up.res3.0.shortcut.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight} | (512,) (512,) (512,) (512,) (512,256,1,1)       |
| backbone.bottom_up.res3.1.conv1.*                | backbone.bottom_up.res3.1.conv1.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (128,) (128,) (128,) (128,) (128,512,1,1)       |
| backbone.bottom_up.res3.1.conv2.*                | backbone.bottom_up.res3.1.conv2.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (128,) (128,) (128,) (128,) (128,128,3,3)       |
| backbone.bottom_up.res3.1.conv3.*                | backbone.bottom_up.res3.1.conv3.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (512,) (512,) (512,) (512,) (512,128,1,1)       |
| backbone.bottom_up.res3.2.conv1.*                | backbone.bottom_up.res3.2.conv1.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (128,) (128,) (128,) (128,) (128,512,1,1)       |
| backbone.bottom_up.res3.2.conv2.*                | backbone.bottom_up.res3.2.conv2.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (128,) (128,) (128,) (128,) (128,128,3,3)       |
| backbone.bottom_up.res3.2.conv3.*                | backbone.bottom_up.res3.2.conv3.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (512,) (512,) (512,) (512,) (512,128,1,1)       |
| backbone.bottom_up.res3.3.conv1.*                | backbone.bottom_up.res3.3.conv1.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (128,) (128,) (128,) (128,) (128,512,1,1)       |
| backbone.bottom_up.res3.3.conv2.*                | backbone.bottom_up.res3.3.conv2.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (128,) (128,) (128,) (128,) (128,128,3,3)       |
| backbone.bottom_up.res3.3.conv3.*                | backbone.bottom_up.res3.3.conv3.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (512,) (512,) (512,) (512,) (512,128,1,1)       |
| backbone.bottom_up.res4.0.conv1.*                | backbone.bottom_up.res4.0.conv1.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (256,) (256,) (256,) (256,) (256,512,1,1)       |
| backbone.bottom_up.res4.0.conv2.*                | backbone.bottom_up.res4.0.conv2.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (256,) (256,) (256,) (256,) (256,256,3,3)       |
| backbone.bottom_up.res4.0.conv3.*                | backbone.bottom_up.res4.0.conv3.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (1024,) (1024,) (1024,) (1024,) (1024,256,1,1)  |
| backbone.bottom_up.res4.0.shortcut.*             | backbone.bottom_up.res4.0.shortcut.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight} | (1024,) (1024,) (1024,) (1024,) (1024,512,1,1)  |
| backbone.bottom_up.res4.1.conv1.*                | backbone.bottom_up.res4.1.conv1.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (256,) (256,) (256,) (256,) (256,1024,1,1)      |
| backbone.bottom_up.res4.1.conv2.*                | backbone.bottom_up.res4.1.conv2.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (256,) (256,) (256,) (256,) (256,256,3,3)       |
| backbone.bottom_up.res4.1.conv3.*                | backbone.bottom_up.res4.1.conv3.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (1024,) (1024,) (1024,) (1024,) (1024,256,1,1)  |
| backbone.bottom_up.res4.2.conv1.*                | backbone.bottom_up.res4.2.conv1.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (256,) (256,) (256,) (256,) (256,1024,1,1)      |
| backbone.bottom_up.res4.2.conv2.*                | backbone.bottom_up.res4.2.conv2.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (256,) (256,) (256,) (256,) (256,256,3,3)       |
| backbone.bottom_up.res4.2.conv3.*                | backbone.bottom_up.res4.2.conv3.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (1024,) (1024,) (1024,) (1024,) (1024,256,1,1)  |
| backbone.bottom_up.res4.3.conv1.*                | backbone.bottom_up.res4.3.conv1.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (256,) (256,) (256,) (256,) (256,1024,1,1)      |
| backbone.bottom_up.res4.3.conv2.*                | backbone.bottom_up.res4.3.conv2.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (256,) (256,) (256,) (256,) (256,256,3,3)       |
| backbone.bottom_up.res4.3.conv3.*                | backbone.bottom_up.res4.3.conv3.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (1024,) (1024,) (1024,) (1024,) (1024,256,1,1)  |
| backbone.bottom_up.res4.4.conv1.*                | backbone.bottom_up.res4.4.conv1.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (256,) (256,) (256,) (256,) (256,1024,1,1)      |
| backbone.bottom_up.res4.4.conv2.*                | backbone.bottom_up.res4.4.conv2.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (256,) (256,) (256,) (256,) (256,256,3,3)       |
| backbone.bottom_up.res4.4.conv3.*                | backbone.bottom_up.res4.4.conv3.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (1024,) (1024,) (1024,) (1024,) (1024,256,1,1)  |
| backbone.bottom_up.res4.5.conv1.*                | backbone.bottom_up.res4.5.conv1.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (256,) (256,) (256,) (256,) (256,1024,1,1)      |
| backbone.bottom_up.res4.5.conv2.*                | backbone.bottom_up.res4.5.conv2.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (256,) (256,) (256,) (256,) (256,256,3,3)       |
| backbone.bottom_up.res4.5.conv3.*                | backbone.bottom_up.res4.5.conv3.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (1024,) (1024,) (1024,) (1024,) (1024,256,1,1)  |
| backbone.bottom_up.res5.0.conv1.*                | backbone.bottom_up.res5.0.conv1.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (512,) (512,) (512,) (512,) (512,1024,1,1)      |
| backbone.bottom_up.res5.0.conv2.*                | backbone.bottom_up.res5.0.conv2.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (512,) (512,) (512,) (512,) (512,512,3,3)       |
| backbone.bottom_up.res5.0.conv3.*                | backbone.bottom_up.res5.0.conv3.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (2048,) (2048,) (2048,) (2048,) (2048,512,1,1)  |
| backbone.bottom_up.res5.0.shortcut.*             | backbone.bottom_up.res5.0.shortcut.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight} | (2048,) (2048,) (2048,) (2048,) (2048,1024,1,1) |
| backbone.bottom_up.res5.1.conv1.*                | backbone.bottom_up.res5.1.conv1.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (512,) (512,) (512,) (512,) (512,2048,1,1)      |
| backbone.bottom_up.res5.1.conv2.*                | backbone.bottom_up.res5.1.conv2.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (512,) (512,) (512,) (512,) (512,512,3,3)       |
| backbone.bottom_up.res5.1.conv3.*                | backbone.bottom_up.res5.1.conv3.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (2048,) (2048,) (2048,) (2048,) (2048,512,1,1)  |
| backbone.bottom_up.res5.2.conv1.*                | backbone.bottom_up.res5.2.conv1.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (512,) (512,) (512,) (512,) (512,2048,1,1)      |
| backbone.bottom_up.res5.2.conv2.*                | backbone.bottom_up.res5.2.conv2.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (512,) (512,) (512,) (512,) (512,512,3,3)       |
| backbone.bottom_up.res5.2.conv3.*                | backbone.bottom_up.res5.2.conv3.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (2048,) (2048,) (2048,) (2048,) (2048,512,1,1)  |
| backbone.bottom_up.stem.conv1.*                  | backbone.bottom_up.stem.conv1.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}      | (64,) (64,) (64,) (64,) (64,3,7,7)              |
| backbone.fpn_lateral2.*                          | backbone.fpn_lateral2.{bias,weight}                                                                  | (256,) (256,256,1,1)                            |
| backbone.fpn_lateral3.*                          | backbone.fpn_lateral3.{bias,weight}                                                                  | (256,) (256,512,1,1)                            |
| backbone.fpn_lateral4.*                          | backbone.fpn_lateral4.{bias,weight}                                                                  | (256,) (256,1024,1,1)                           |
| backbone.fpn_lateral5.*                          | backbone.fpn_lateral5.{bias,weight}                                                                  | (256,) (256,2048,1,1)                           |
| backbone.fpn_output2.*                           | backbone.fpn_output2.{bias,weight}                                                                   | (256,) (256,256,3,3)                            |
| backbone.fpn_output3.*                           | backbone.fpn_output3.{bias,weight}                                                                   | (256,) (256,256,3,3)                            |
| backbone.fpn_output4.*                           | backbone.fpn_output4.{bias,weight}                                                                   | (256,) (256,256,3,3)                            |
| backbone.fpn_output5.*                           | backbone.fpn_output5.{bias,weight}                                                                   | (256,) (256,256,3,3)                            |
| betas                                            | betas                                                                                                | (1000,)                                         |
| head.head_series.0.bboxes_delta.*                | head.head_series.0.bboxes_delta.{bias,weight}                                                        | (4,) (4,256)                                    |
| head.head_series.0.block_time_mlp.1.*            | head.head_series.0.block_time_mlp.1.{bias,weight}                                                    | (512,) (512,1024)                               |
| head.head_series.0.class_logits.*                | head.head_series.0.class_logits.{bias,weight}                                                        | (80,) (80,256)                                  |
| head.head_series.0.cls_module.0.weight           | head.head_series.0.cls_module.0.weight                                                               | (256, 256)                                      |
| head.head_series.0.cls_module.1.*                | head.head_series.0.cls_module.1.{bias,weight}                                                        | (256,) (256,)                                   |
| head.head_series.0.inst_interact.dynamic_layer.* | head.head_series.0.inst_interact.dynamic_layer.{bias,weight}                                         | (32768,) (32768,256)                            |
| head.head_series.0.inst_interact.norm1.*         | head.head_series.0.inst_interact.norm1.{bias,weight}                                                 | (64,) (64,)                                     |
| head.head_series.0.inst_interact.norm2.*         | head.head_series.0.inst_interact.norm2.{bias,weight}                                                 | (256,) (256,)                                   |
| head.head_series.0.inst_interact.norm3.*         | head.head_series.0.inst_interact.norm3.{bias,weight}                                                 | (256,) (256,)                                   |
| head.head_series.0.inst_interact.out_layer.*     | head.head_series.0.inst_interact.out_layer.{bias,weight}                                             | (256,) (256,12544)                              |
| head.head_series.0.linear1.*                     | head.head_series.0.linear1.{bias,weight}                                                             | (2048,) (2048,256)                              |
| head.head_series.0.linear2.*                     | head.head_series.0.linear2.{bias,weight}                                                             | (256,) (256,2048)                               |
| head.head_series.0.norm1.*                       | head.head_series.0.norm1.{bias,weight}                                                               | (256,) (256,)                                   |
| head.head_series.0.norm2.*                       | head.head_series.0.norm2.{bias,weight}                                                               | (256,) (256,)                                   |
| head.head_series.0.norm3.*                       | head.head_series.0.norm3.{bias,weight}                                                               | (256,) (256,)                                   |
| head.head_series.0.reg_module.0.weight           | head.head_series.0.reg_module.0.weight                                                               | (256, 256)                                      |
| head.head_series.0.reg_module.1.*                | head.head_series.0.reg_module.1.{bias,weight}                                                        | (256,) (256,)                                   |
| head.head_series.0.reg_module.3.weight           | head.head_series.0.reg_module.3.weight                                                               | (256, 256)                                      |
| head.head_series.0.reg_module.4.*                | head.head_series.0.reg_module.4.{bias,weight}                                                        | (256,) (256,)                                   |
| head.head_series.0.reg_module.6.weight           | head.head_series.0.reg_module.6.weight                                                               | (256, 256)                                      |
| head.head_series.0.reg_module.7.*                | head.head_series.0.reg_module.7.{bias,weight}                                                        | (256,) (256,)                                   |
| head.head_series.0.self_attn.*                   | head.head_series.0.self_attn.{in_proj_bias,in_proj_weight,out_proj.bias,out_proj.weight}             | (768,) (768,256) (256,) (256,256)               |
| head.head_series.1.bboxes_delta.*                | head.head_series.1.bboxes_delta.{bias,weight}                                                        | (4,) (4,256)                                    |
| head.head_series.1.block_time_mlp.1.*            | head.head_series.1.block_time_mlp.1.{bias,weight}                                                    | (512,) (512,1024)                               |
| head.head_series.1.class_logits.*                | head.head_series.1.class_logits.{bias,weight}                                                        | (80,) (80,256)                                  |
| head.head_series.1.cls_module.0.weight           | head.head_series.1.cls_module.0.weight                                                               | (256, 256)                                      |
| head.head_series.1.cls_module.1.*                | head.head_series.1.cls_module.1.{bias,weight}                                                        | (256,) (256,)                                   |
| head.head_series.1.inst_interact.dynamic_layer.* | head.head_series.1.inst_interact.dynamic_layer.{bias,weight}                                         | (32768,) (32768,256)                            |
| head.head_series.1.inst_interact.norm1.*         | head.head_series.1.inst_interact.norm1.{bias,weight}                                                 | (64,) (64,)                                     |
| head.head_series.1.inst_interact.norm2.*         | head.head_series.1.inst_interact.norm2.{bias,weight}                                                 | (256,) (256,)                                   |
| head.head_series.1.inst_interact.norm3.*         | head.head_series.1.inst_interact.norm3.{bias,weight}                                                 | (256,) (256,)                                   |
| head.head_series.1.inst_interact.out_layer.*     | head.head_series.1.inst_interact.out_layer.{bias,weight}                                             | (256,) (256,12544)                              |
| head.head_series.1.linear1.*                     | head.head_series.1.linear1.{bias,weight}                                                             | (2048,) (2048,256)                              |
| head.head_series.1.linear2.*                     | head.head_series.1.linear2.{bias,weight}                                                             | (256,) (256,2048)                               |
| head.head_series.1.norm1.*                       | head.head_series.1.norm1.{bias,weight}                                                               | (256,) (256,)                                   |
| head.head_series.1.norm2.*                       | head.head_series.1.norm2.{bias,weight}                                                               | (256,) (256,)                                   |
| head.head_series.1.norm3.*                       | head.head_series.1.norm3.{bias,weight}                                                               | (256,) (256,)                                   |
| head.head_series.1.reg_module.0.weight           | head.head_series.1.reg_module.0.weight                                                               | (256, 256)                                      |
| head.head_series.1.reg_module.1.*                | head.head_series.1.reg_module.1.{bias,weight}                                                        | (256,) (256,)                                   |
| head.head_series.1.reg_module.3.weight           | head.head_series.1.reg_module.3.weight                                                               | (256, 256)                                      |
| head.head_series.1.reg_module.4.*                | head.head_series.1.reg_module.4.{bias,weight}                                                        | (256,) (256,)                                   |
| head.head_series.1.reg_module.6.weight           | head.head_series.1.reg_module.6.weight                                                               | (256, 256)                                      |
| head.head_series.1.reg_module.7.*                | head.head_series.1.reg_module.7.{bias,weight}                                                        | (256,) (256,)                                   |
| head.head_series.1.self_attn.*                   | head.head_series.1.self_attn.{in_proj_bias,in_proj_weight,out_proj.bias,out_proj.weight}             | (768,) (768,256) (256,) (256,256)               |
| head.head_series.2.bboxes_delta.*                | head.head_series.2.bboxes_delta.{bias,weight}                                                        | (4,) (4,256)                                    |
| head.head_series.2.block_time_mlp.1.*            | head.head_series.2.block_time_mlp.1.{bias,weight}                                                    | (512,) (512,1024)                               |
| head.head_series.2.class_logits.*                | head.head_series.2.class_logits.{bias,weight}                                                        | (80,) (80,256)                                  |
| head.head_series.2.cls_module.0.weight           | head.head_series.2.cls_module.0.weight                                                               | (256, 256)                                      |
| head.head_series.2.cls_module.1.*                | head.head_series.2.cls_module.1.{bias,weight}                                                        | (256,) (256,)                                   |
| head.head_series.2.inst_interact.dynamic_layer.* | head.head_series.2.inst_interact.dynamic_layer.{bias,weight}                                         | (32768,) (32768,256)                            |
| head.head_series.2.inst_interact.norm1.*         | head.head_series.2.inst_interact.norm1.{bias,weight}                                                 | (64,) (64,)                                     |
| head.head_series.2.inst_interact.norm2.*         | head.head_series.2.inst_interact.norm2.{bias,weight}                                                 | (256,) (256,)                                   |
| head.head_series.2.inst_interact.norm3.*         | head.head_series.2.inst_interact.norm3.{bias,weight}                                                 | (256,) (256,)                                   |
| head.head_series.2.inst_interact.out_layer.*     | head.head_series.2.inst_interact.out_layer.{bias,weight}                                             | (256,) (256,12544)                              |
| head.head_series.2.linear1.*                     | head.head_series.2.linear1.{bias,weight}                                                             | (2048,) (2048,256)                              |
| head.head_series.2.linear2.*                     | head.head_series.2.linear2.{bias,weight}                                                             | (256,) (256,2048)                               |
| head.head_series.2.norm1.*                       | head.head_series.2.norm1.{bias,weight}                                                               | (256,) (256,)                                   |
| head.head_series.2.norm2.*                       | head.head_series.2.norm2.{bias,weight}                                                               | (256,) (256,)                                   |
| head.head_series.2.norm3.*                       | head.head_series.2.norm3.{bias,weight}                                                               | (256,) (256,)                                   |
| head.head_series.2.reg_module.0.weight           | head.head_series.2.reg_module.0.weight                                                               | (256, 256)                                      |
| head.head_series.2.reg_module.1.*                | head.head_series.2.reg_module.1.{bias,weight}                                                        | (256,) (256,)                                   |
| head.head_series.2.reg_module.3.weight           | head.head_series.2.reg_module.3.weight                                                               | (256, 256)                                      |
| head.head_series.2.reg_module.4.*                | head.head_series.2.reg_module.4.{bias,weight}                                                        | (256,) (256,)                                   |
| head.head_series.2.reg_module.6.weight           | head.head_series.2.reg_module.6.weight                                                               | (256, 256)                                      |
| head.head_series.2.reg_module.7.*                | head.head_series.2.reg_module.7.{bias,weight}                                                        | (256,) (256,)                                   |
| head.head_series.2.self_attn.*                   | head.head_series.2.self_attn.{in_proj_bias,in_proj_weight,out_proj.bias,out_proj.weight}             | (768,) (768,256) (256,) (256,256)               |
| head.head_series.3.bboxes_delta.*                | head.head_series.3.bboxes_delta.{bias,weight}                                                        | (4,) (4,256)                                    |
| head.head_series.3.block_time_mlp.1.*            | head.head_series.3.block_time_mlp.1.{bias,weight}                                                    | (512,) (512,1024)                               |
| head.head_series.3.class_logits.*                | head.head_series.3.class_logits.{bias,weight}                                                        | (80,) (80,256)                                  |
| head.head_series.3.cls_module.0.weight           | head.head_series.3.cls_module.0.weight                                                               | (256, 256)                                      |
| head.head_series.3.cls_module.1.*                | head.head_series.3.cls_module.1.{bias,weight}                                                        | (256,) (256,)                                   |
| head.head_series.3.inst_interact.dynamic_layer.* | head.head_series.3.inst_interact.dynamic_layer.{bias,weight}                                         | (32768,) (32768,256)                            |
| head.head_series.3.inst_interact.norm1.*         | head.head_series.3.inst_interact.norm1.{bias,weight}                                                 | (64,) (64,)                                     |
| head.head_series.3.inst_interact.norm2.*         | head.head_series.3.inst_interact.norm2.{bias,weight}                                                 | (256,) (256,)                                   |
| head.head_series.3.inst_interact.norm3.*         | head.head_series.3.inst_interact.norm3.{bias,weight}                                                 | (256,) (256,)                                   |
| head.head_series.3.inst_interact.out_layer.*     | head.head_series.3.inst_interact.out_layer.{bias,weight}                                             | (256,) (256,12544)                              |
| head.head_series.3.linear1.*                     | head.head_series.3.linear1.{bias,weight}                                                             | (2048,) (2048,256)                              |
| head.head_series.3.linear2.*                     | head.head_series.3.linear2.{bias,weight}                                                             | (256,) (256,2048)                               |
| head.head_series.3.norm1.*                       | head.head_series.3.norm1.{bias,weight}                                                               | (256,) (256,)                                   |
| head.head_series.3.norm2.*                       | head.head_series.3.norm2.{bias,weight}                                                               | (256,) (256,)                                   |
| head.head_series.3.norm3.*                       | head.head_series.3.norm3.{bias,weight}                                                               | (256,) (256,)                                   |
| head.head_series.3.reg_module.0.weight           | head.head_series.3.reg_module.0.weight                                                               | (256, 256)                                      |
| head.head_series.3.reg_module.1.*                | head.head_series.3.reg_module.1.{bias,weight}                                                        | (256,) (256,)                                   |
| head.head_series.3.reg_module.3.weight           | head.head_series.3.reg_module.3.weight                                                               | (256, 256)                                      |
| head.head_series.3.reg_module.4.*                | head.head_series.3.reg_module.4.{bias,weight}                                                        | (256,) (256,)                                   |
| head.head_series.3.reg_module.6.weight           | head.head_series.3.reg_module.6.weight                                                               | (256, 256)                                      |
| head.head_series.3.reg_module.7.*                | head.head_series.3.reg_module.7.{bias,weight}                                                        | (256,) (256,)                                   |
| head.head_series.3.self_attn.*                   | head.head_series.3.self_attn.{in_proj_bias,in_proj_weight,out_proj.bias,out_proj.weight}             | (768,) (768,256) (256,) (256,256)               |
| head.head_series.4.bboxes_delta.*                | head.head_series.4.bboxes_delta.{bias,weight}                                                        | (4,) (4,256)                                    |
| head.head_series.4.block_time_mlp.1.*            | head.head_series.4.block_time_mlp.1.{bias,weight}                                                    | (512,) (512,1024)                               |
| head.head_series.4.class_logits.*                | head.head_series.4.class_logits.{bias,weight}                                                        | (80,) (80,256)                                  |
| head.head_series.4.cls_module.0.weight           | head.head_series.4.cls_module.0.weight                                                               | (256, 256)                                      |
| head.head_series.4.cls_module.1.*                | head.head_series.4.cls_module.1.{bias,weight}                                                        | (256,) (256,)                                   |
| head.head_series.4.inst_interact.dynamic_layer.* | head.head_series.4.inst_interact.dynamic_layer.{bias,weight}                                         | (32768,) (32768,256)                            |
| head.head_series.4.inst_interact.norm1.*         | head.head_series.4.inst_interact.norm1.{bias,weight}                                                 | (64,) (64,)                                     |
| head.head_series.4.inst_interact.norm2.*         | head.head_series.4.inst_interact.norm2.{bias,weight}                                                 | (256,) (256,)                                   |
| head.head_series.4.inst_interact.norm3.*         | head.head_series.4.inst_interact.norm3.{bias,weight}                                                 | (256,) (256,)                                   |
| head.head_series.4.inst_interact.out_layer.*     | head.head_series.4.inst_interact.out_layer.{bias,weight}                                             | (256,) (256,12544)                              |
| head.head_series.4.linear1.*                     | head.head_series.4.linear1.{bias,weight}                                                             | (2048,) (2048,256)                              |
| head.head_series.4.linear2.*                     | head.head_series.4.linear2.{bias,weight}                                                             | (256,) (256,2048)                               |
| head.head_series.4.norm1.*                       | head.head_series.4.norm1.{bias,weight}                                                               | (256,) (256,)                                   |
| head.head_series.4.norm2.*                       | head.head_series.4.norm2.{bias,weight}                                                               | (256,) (256,)                                   |
| head.head_series.4.norm3.*                       | head.head_series.4.norm3.{bias,weight}                                                               | (256,) (256,)                                   |
| head.head_series.4.reg_module.0.weight           | head.head_series.4.reg_module.0.weight                                                               | (256, 256)                                      |
| head.head_series.4.reg_module.1.*                | head.head_series.4.reg_module.1.{bias,weight}                                                        | (256,) (256,)                                   |
| head.head_series.4.reg_module.3.weight           | head.head_series.4.reg_module.3.weight                                                               | (256, 256)                                      |
| head.head_series.4.reg_module.4.*                | head.head_series.4.reg_module.4.{bias,weight}                                                        | (256,) (256,)                                   |
| head.head_series.4.reg_module.6.weight           | head.head_series.4.reg_module.6.weight                                                               | (256, 256)                                      |
| head.head_series.4.reg_module.7.*                | head.head_series.4.reg_module.7.{bias,weight}                                                        | (256,) (256,)                                   |
| head.head_series.4.self_attn.*                   | head.head_series.4.self_attn.{in_proj_bias,in_proj_weight,out_proj.bias,out_proj.weight}             | (768,) (768,256) (256,) (256,256)               |
| head.head_series.5.bboxes_delta.*                | head.head_series.5.bboxes_delta.{bias,weight}                                                        | (4,) (4,256)                                    |
| head.head_series.5.block_time_mlp.1.*            | head.head_series.5.block_time_mlp.1.{bias,weight}                                                    | (512,) (512,1024)                               |
| head.head_series.5.class_logits.*                | head.head_series.5.class_logits.{bias,weight}                                                        | (80,) (80,256)                                  |
| head.head_series.5.cls_module.0.weight           | head.head_series.5.cls_module.0.weight                                                               | (256, 256)                                      |
| head.head_series.5.cls_module.1.*                | head.head_series.5.cls_module.1.{bias,weight}                                                        | (256,) (256,)                                   |
| head.head_series.5.inst_interact.dynamic_layer.* | head.head_series.5.inst_interact.dynamic_layer.{bias,weight}                                         | (32768,) (32768,256)                            |
| head.head_series.5.inst_interact.norm1.*         | head.head_series.5.inst_interact.norm1.{bias,weight}                                                 | (64,) (64,)                                     |
| head.head_series.5.inst_interact.norm2.*         | head.head_series.5.inst_interact.norm2.{bias,weight}                                                 | (256,) (256,)                                   |
| head.head_series.5.inst_interact.norm3.*         | head.head_series.5.inst_interact.norm3.{bias,weight}                                                 | (256,) (256,)                                   |
| head.head_series.5.inst_interact.out_layer.*     | head.head_series.5.inst_interact.out_layer.{bias,weight}                                             | (256,) (256,12544)                              |
| head.head_series.5.linear1.*                     | head.head_series.5.linear1.{bias,weight}                                                             | (2048,) (2048,256)                              |
| head.head_series.5.linear2.*                     | head.head_series.5.linear2.{bias,weight}                                                             | (256,) (256,2048)                               |
| head.head_series.5.norm1.*                       | head.head_series.5.norm1.{bias,weight}                                                               | (256,) (256,)                                   |
| head.head_series.5.norm2.*                       | head.head_series.5.norm2.{bias,weight}                                                               | (256,) (256,)                                   |
| head.head_series.5.norm3.*                       | head.head_series.5.norm3.{bias,weight}                                                               | (256,) (256,)                                   |
| head.head_series.5.reg_module.0.weight           | head.head_series.5.reg_module.0.weight                                                               | (256, 256)                                      |
| head.head_series.5.reg_module.1.*                | head.head_series.5.reg_module.1.{bias,weight}                                                        | (256,) (256,)                                   |
| head.head_series.5.reg_module.3.weight           | head.head_series.5.reg_module.3.weight                                                               | (256, 256)                                      |
| head.head_series.5.reg_module.4.*                | head.head_series.5.reg_module.4.{bias,weight}                                                        | (256,) (256,)                                   |
| head.head_series.5.reg_module.6.weight           | head.head_series.5.reg_module.6.weight                                                               | (256, 256)                                      |
| head.head_series.5.reg_module.7.*                | head.head_series.5.reg_module.7.{bias,weight}                                                        | (256,) (256,)                                   |
| head.head_series.5.self_attn.*                   | head.head_series.5.self_attn.{in_proj_bias,in_proj_weight,out_proj.bias,out_proj.weight}             | (768,) (768,256) (256,) (256,256)               |
| head.time_mlp.1.*                                | head.time_mlp.1.{bias,weight}                                                                        | (1024,) (1024,256)                              |
| head.time_mlp.3.*                                | head.time_mlp.3.{bias,weight}                                                                        | (1024,) (1024,1024)                             |
| log_one_minus_alphas_cumprod                     | log_one_minus_alphas_cumprod                                                                         | (1000,)                                         |
| posterior_log_variance_clipped                   | posterior_log_variance_clipped                                                                       | (1000,)                                         |
| posterior_mean_coef1                             | posterior_mean_coef1                                                                                 | (1000,)                                         |
| posterior_mean_coef2                             | posterior_mean_coef2                                                                                 | (1000,)                                         |
| posterior_variance                               | posterior_variance                                                                                   | (1000,)                                         |
| sqrt_alphas_cumprod                              | sqrt_alphas_cumprod                                                                                  | (1000,)                                         |
| sqrt_one_minus_alphas_cumprod                    | sqrt_one_minus_alphas_cumprod                                                                        | (1000,)                                         |
| sqrt_recip_alphas_cumprod                        | sqrt_recip_alphas_cumprod                                                                            | (1000,)                                         |
| sqrt_recipm1_alphas_cumprod                      | sqrt_recipm1_alphas_cumprod                                                                          | (1000,)                                         |
[11/25 00:30:34 detectron2]: image.jpg: detected 1 instances in 8.20s
```

<br>

Opencv window pops up with the output image with the bounding box. 

<br>





Thanks for the interest to learn something new. Please reach out in case if you face any difficulties. Always happy to help. 