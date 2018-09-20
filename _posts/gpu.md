---
layout: post
title: "GPU with Tensorflow"
author: "Karthik"
categories: journal
tags: [documentation,sample]
<!-- image: cards.jpg -->
---


# What is GPU? 
---
Graphical Processing unit is a electronic circuit designed to perform rapidly executions than CPUs.Modern GPUs are very efficient at manipulating computer graphics and image processing, and their highly parallel structure makes them more efficient than general purpose CPUs for algorithms where the processing of large blocks of data is done in parallel. 


# When to use GPU ?
---
In general, if the step of the process can be described such as “do this mathematical operation thousands of times”, then send it to the GPU. Examples include matrix multiplication and computing the inverse of a matrix. In fact, many basic matrix operations are prime candidates for GPUs. As an overly broad and simple rule, other operations should be performed on the CPU.
There is also a cost to changing devices and using GPUs. GPUs don’t have direct access to the rest of your computer (except, of course for the display). Due to this, if you are running a command on a GPU, you need to copy all of the data to the GPU first, then do the operation, then copy the result back to your computer’s main memory. TensorFlow handles this under the hood, so the code is simple, but the work still needs to be performed.


# GPU setting in tensorflow vs PyTorch
---
Device management in TensorFlow is about as seamless as it gets. Usually you don’t need to specify anything since the defaults are set well. For example, TensorFlow assumes you want to run on the GPU if one is available.

In PyTorch you have to explicitly move everything onto the device even if CUDA is enabled.

**NOTE:**  The downside with TensorFlow device management is that by default it consumes all the memory on all available GPUs even if only one is being used. The simple workaround is to specify **CUDA_VISIBLE_DEVICES**. Sometimes people forget this, and GPUs can appear to be busy when they are in fact idle.


**SOLUTION:**
You can set environment variables in the notebook using os.environ. Do the following before initializing TensorFlow to limit TensorFlow to first GPU.

```python
import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0"
```
[How to get "PCI_BUS_ID"](https://wiki.debian.org/HowToIdentifyADevice/PCI)


### You can double check that you have the correct devices visible to TF

```python
from tensorflow.python.client 
import device_lib 
print(device_lib.list_local_devices())
```


# Tensorflow Supported devices

On a typical system, there are multiple computing devices. In TensorFlow, the supported device types are CPU and GPU. They are represented as strings. For example:

```python
"/cpu:0": The CPU of your machine.
"/device:GPU:0": The GPU of your machine, if you have one.
"/device:GPU:1": The second GPU of your machine, etc.
```

> If a TensorFlow operation has both CPU and GPU implementations, the GPU devices will be given priority when the operation is assigned to a device.


# Manual device placement

If you would like a particular operation to run on a device of your choice instead of what's automatically selected for you, you can use 
_**with tf.device**_ to create a device context such that all the operations within that context will have the same device assignment.
 
# Allowing GPU memory growth

By default, TensorFlow maps nearly all of the GPU memory of all GPUs (subject to CUDA_VISIBLE_DEVICES) visible to the process. This is done to more efficiently use the relatively precious GPU memory resources on the devices by reducing memory fragmentation.
In some cases it is desirable for the process to only allocate a subset of the available memory, or to only grow the memory usage as is needed by the process. TensorFlow provides two Config options on the Session to control this.
The first is the allow_growth option, which attempts to allocate only as much GPU memory based on runtime allocations: it starts out allocating very little memory, and as Sessions get run and more GPU memory is needed, we extend the GPU memory region needed by the TensorFlow process. Note that we do not release memory, since that can lead to even worse memory fragmentation. To turn this option on, set the option in the **ConfigProto** by:

```python
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config, ...)
```


# Using GPU if available else using CPU

```python
if tf.test.is_gpu_available() == True: processor = '/gpu:0'
else : processor = '/cpu:0'
 with tf.device(processor):

```
