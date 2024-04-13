---
layout: post
title: "tsar model inference"
author: "Karthik"
categories: journal
tags: [documentation,sample]



---





# Introduction:

tsai is an open source deep learning package built on top of Pytorch and fastai focused on tasks like classification, regression, forecasting, imputation and many more. 
The aim of the blog post is to provide solution in performing inference of tsai trained classification model. The training method is simple, but I was facing problem in inferencing because of the version issue. I want to write a detailed blog on resolving and inferencing using the trained model.

## Installation

```
Python version 3.8.18

pip install tsai==0.3.9
pip install torch==2.2.0
pip install matplotlib==3.7.1 (This version should match or the inference load_learner will throw error)
``` 


## Data preparation

The data that I am using here is a time series classification task. This is the data represented in a table format.

| Feature1 | Feature2 | Feature3 | Label |
|----------|----------|----------|-------|
| 1        | 56       | 67       | 1     |
| 2        | 67       | 33       | 2     |
| 3        | 45       | 21       | 1     |


```
from sklearn.model_selection import train_test_split
import pandas as pd

''' loading the csv file '''
df = pd.read_csv('/content/detrend_df.csv')

''' spliting the features and labels into seperate variable '''
X = df.iloc[:, :200].values
y = df['label'].values

''' splitting data '''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

''' reshaping '''
X_train = X_train.reshape((X_train.shape[0], 1, X.shape[1]))
y_train = y_train.reshape((y_train.shape[0], 1))

X_test = X_test.reshape((X_test.shape[0], 1, X.shape[1]))
y_test = y_test.reshape((y_test.shape[0], 1))

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
```


## Training

```
from tsai.all import *
import numpy as np


tfms = [None, TSClassification()]
batch_tfms = TSStandardize()

clf = TSClassifier(X_train,
                   y_train,
                   path='models',
                   arch="InceptionTimePlus",
                   tfms=tfms,
                   batch_tfms=batch_tfms,
                   metrics=accuracy,
                   cbs=ShowGraph(),
                   train_metrics=True)

clf.fit_one_cycle(100, 3e-4)

clf.export("clf.pkl")
```


## Inference

In this section I will share the issues that I faced while performing inference with the trained exported model. 


#### Error

```
from tsai.inference import load_learner

learn = load_learner("clf.pkl")
```

```
/Users/karthikrajedran/anaconda3/envs/segmentationModel/lib/python3.8/site-packages/transformers/utils/generic.py:441: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
  _torch_pytree._register_pytree_node(
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/Users/karthikrajedran/anaconda3/envs/segmentationModel/lib/python3.8/site-packages/fastai/learner.py", line 446, in load_learner
    try: res = torch.load(fname, map_location=map_loc, pickle_module=pickle_module)
  File "/Users/karthikrajedran/anaconda3/envs/segmentationModel/lib/python3.8/site-packages/torch/serialization.py", line 1026, in load
    return _load(opened_zipfile,
  File "/Users/karthikrajedran/anaconda3/envs/segmentationModel/lib/python3.8/site-packages/torch/serialization.py", line 1438, in _load
    result = unpickler.load()
  File "/Users/karthikrajedran/anaconda3/envs/segmentationModel/lib/python3.8/site-packages/matplotlib/cbook/__init__.py", line 229, in __setstate__
    self._cid_gen = itertools.count(cid_count)
TypeError: a number is required
```

##### Try 1: Load directly with torch

```
import torch

torch.load("clf.pkl", map_location=torch.device('cpu'))
```

```
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/Users/karthikrajedran/anaconda3/envs/segmentationModel/lib/python3.8/site-packages/torch/serialization.py", line 1026, in load
    return _load(opened_zipfile,
  File "/Users/karthikrajedran/anaconda3/envs/segmentationModel/lib/python3.8/site-packages/torch/serialization.py", line 1438, in _load
    result = unpickler.load()
  File "/Users/karthikrajedran/anaconda3/envs/segmentationModel/lib/python3.8/site-packages/matplotlib/cbook/__init__.py", line 229, in __setstate__
    self._cid_gen = itertools.count(cid_count)
TypeError: a number is required
```

##### Try 2: From the error logs, it shows that the error was raised from the matplotlib library. Hence I changed the matplotlib version to 3.7.1 after some research. 

```
pip install matplotlib==3.7.1
```
Then I tried the inference method

```
from tsai.inference import load_learner

learn = load_learner("clf.pkl")
```

Then the model was loaded successfully.


##### Alternative idea:
I wanted to save the learner after training in a pickle format and load it for inference. But this is an inefficient method.  

## Summary

This is a blog post to show on how I fixed the inference issue in tsai package. There were no solid solution to fix this problem, hence I am sharing the method that I followed to fix this issue in this blog post. 

