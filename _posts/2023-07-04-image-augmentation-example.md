---
layout: post
title: "Image augmentation with imgaug"
author: "Karthik"
categories: journal
tags: [documentation,sample]






---



This is a code snippet to generate image augmentation on your training data. 



```
from imgaug import augmenters as iaa
from skimage.io import imread_collection
import cv2
import numpy as np

# list .PNG files from the folder
IMAGE_FOLDER_PATH = "/content/*.PNG"


seq = iaa.Sequential([
    iaa.Crop(px=(0, 16)), # crop images from each side by 0 to 16px (randomly chosen)
    iaa.Fliplr(0.5), # horizontally flip 50% of the images
    iaa.GaussianBlur(sigma=(0, 3.0)) # blur images with a sigma of 0 to 3.0
])


# number of times to run the augmentation
for batch_idx in range(1):
	# load all the .PNG images from a folder 
    images = imread_collection(IMAGE_FOLDER_PATH)
    images_aug = seq(images=images)
    for each_aug_image in images_aug:
    	# write the augmented image in a folder with random integer filename 
    	cv2.imwrite(f"/content/sample_data/x/{np.random.randint(9999)}.png", each_aug_image)
    
```

