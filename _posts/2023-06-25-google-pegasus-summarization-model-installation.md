---
layout: post
title: "Google pegasus summarization model installation"
author: "Karthik"
categories: journal
tags: [documentation,sample]





---



In this blog post, I explain the steps that I followed to install google pegasus summatization model from the officical github page. I failed to complete the installation, since the pip package "**tensorflow_text**" failed to import, I will mention the error it threw. 

<br>

I cloned the github repository.

<br>

I downloaded the pretrained model weights.  

<br>

Installed the pip dependency packages

<br>

My system configuration is 

```
OS Name: Microsoft Windows 10 Home Single Language
Version: 10.0.19045 Build 19045
System Type: x64-based PC
Processor:	11th Gen Intel(R) Core(TM) i3-1115G4 @ 3.00GHz, 2995 Mhz, 2 Core(s), 4 Logical Processor(s)
```

​	<br>

I will list out the steps that I followed. 

<br>

1. The execution code

```
python3 pegasus/bin/evaluate.py --params=aeslc_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=5,beam_alpha=0.6 --model_dir=ckpt/pegasus_ckpt/aeslc
```

<br>

But the relative import in evaluate.py script did not work, hence I moved the evaluate.py script to root folder. Here the main folder **./pegasus** is my main project folder and all the folders inside are the cloned contents from the repository. 

[![Capture.jpg](https://i.postimg.cc/nzgQhHGM/Capture.jpg)](https://postimg.cc/wy5BWYPd)



<br>

After doing this, the relative import worked and I was able to execute the evaluate.py script. 

<br>

2. pip and python version

```
pip --version
pip 21.2.2 from C:\Users\karth\anaconda3\envs\arxiv_summarizer\lib\site-packages\pip (python 3.6)
```

<br>

I installed the pip packages using ```pip install -r requirements.txt ``` , however it took longer time since pip was looking for compatible version.

<br>

```INFO: pip is looking at multiple versions of portalocker to determine which version is compatible with other requirements. This could take a while.```

<br>

Too avoid this process, I decided to install each pip package individually

```
pip install rouge-score

Name: rouge-score
Version: 0.1.1
Summary: Pure python implementation of ROUGE-1.5.5.
Home-page: https://github.com/google-research/google-research/tree/master/rouge
Author: Google LLC
Author-email: rouge-opensource@google.com
License: UNKNOWN
Location: c:\users\karth\anaconda3\envs\arxiv_summarizer\lib\site-packages
Requires: absl-py, nltk, numpy, six
Required-by:
```

<br>

You need python >= 3.7 for using **tensorflow-text**, since I created the anaconda environment with python 3.6, I upgraded the python version using the command 

```
conda install python=3.7
```

One other requirement is that the version of **tensorflow** and **tensorflow-text** must be same. 

<br>

I installed **tensorflow-text** using the command

```
pip install tensorflow-text
```

```
pip show tensorflow-text

Name: tensorflow-text
Version: 2.6.0
Summary: TF.Text is a TensorFlow library of text related ops, modules, and subgraphs.
Home-page: http://github.com/tensorflow/text
Author: Google Inc.
Author-email: packages@tensorflow.org
License: Apache 2.0
Location: c:\users\karth\anaconda3\envs\arxiv_summarizer\lib\site-packages
Requires: tensorflow, tensorflow-hub
Required-by:
```

<br>

But tensorflow-text failed to import

```
(arxiv_summarizer) C:\Users\karth\OneDrive\Documents\Other_Resources\ProfileProject\pegasus>python evaluate.py --params=aeslc_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=5,beam_alpha=0.6 --model_dir=ckpt/pegasus_ckpt/aeslc

Traceback (most recent call last):
  File "evaluate.py", line 22, in <module>
    from pegasus.data import infeed
  File "C:\Users\karth\OneDrive\Documents\Other_Resources\ProfileProject\pegasus\pegasus\data\infeed.py", line 18, in <module>
    from pegasus.ops import public_parsing_ops
  File "C:\Users\karth\OneDrive\Documents\Other_Resources\ProfileProject\pegasus\pegasus\ops\public_parsing_ops.py", line 23, in <module>
    import tensorflow_text as tf_text
  File "C:\Users\karth\anaconda3\envs\arxiv_summarizer\lib\site-packages\tensorflow_text\__init__.py", line 21, in <module>
    from tensorflow_text.python import metrics
  File "C:\Users\karth\anaconda3\envs\arxiv_summarizer\lib\site-packages\tensorflow_text\python\metrics\__init__.py", line 20, in <module>
    from tensorflow_text.python.metrics.text_similarity_metric_ops import *
  File "C:\Users\karth\anaconda3\envs\arxiv_summarizer\lib\site-packages\tensorflow_text\python\metrics\text_similarity_metric_ops.py", line 28, in <module>
    gen_text_similarity_metric_ops = load_library.load_op_library(resource_loader.get_path_to_datafile('_text_similarity_metric_ops.so'))
  File "C:\Users\karth\AppData\Roaming\Python\Python37\site-packages\tensorflow\python\framework\load_library.py", line 54, in load_op_library
    lib_handle = py_tf.TF_LoadLibrary(library_filename)
tensorflow.python.framework.errors_impl.NotFoundError: C:\Users\karth\anaconda3\envs\arxiv_summarizer\lib\site-packages\tensorflow_text\python\metrics\_text_similarity_metric_ops.so not found

```

<br>

I tried to install the latest version of tensorflow-text 

```
pip install tensorflow-text==2.12.1 
ERROR: Could not find a version that satisfies the requirement tensorflow-text==2.12.1 (from versions: 2.4.0rc0, 2.4.0rc1, 2.4.1, 2.4.2, 2.4.3, 2.5.0rc0, 2.5.0, 2.6.0rc0, 2.6.0, 2.7.0rc0, 2.7.0rc1, 2.7.3, 2.8.0rc0, 2.8.1, 2.8.2, 2.9.0rc0, 2.9.0rc1, 2.9.0, 2.10.0b2, 2.10.0rc0, 2.10.0)
ERROR: No matching distribution found for tensorflow-text==2.12.1
```

Online records said that the latest tensorflow-text pip distribution is not available for windows. 

<br>

I faced issue with PROTOBUF

```
(arxiv_summarizer) C:\Users\karth\OneDrive\Documents\Other_Resources\ProfileProject\pegasus>python evaluate.py --params=aeslc_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=5,beam_alpha=0.6 --model_dir=ckpt/pegasus_ckpt/aeslc

Traceback (most recent call last):
  File "evaluate.py", line 22, in <module>
    from pegasus.data import infeed
  File "C:\Users\karth\OneDrive\Documents\Other_Resources\ProfileProject\pegasus\pegasus\data\infeed.py", line 17, in <module>
    from pegasus.data import all_datasets
  File "C:\Users\karth\OneDrive\Documents\Other_Resources\ProfileProject\pegasus\pegasus\data\all_datasets.py", line 17, in <module>
    from pegasus.data import datasets
  File "C:\Users\karth\OneDrive\Documents\Other_Resources\ProfileProject\pegasus\pegasus\data\datasets.py", line 20, in <module>
    import tensorflow as tf
  File "C:\Users\karth\AppData\Roaming\Python\Python37\site-packages\tensorflow\__init__.py", line 37, in <module>
    from tensorflow.python.tools import module_util as _module_util
  File "C:\Users\karth\AppData\Roaming\Python\Python37\site-packages\tensorflow\python\__init__.py", line 37, in <module>
    from tensorflow.python.eager import context
  File "C:\Users\karth\AppData\Roaming\Python\Python37\site-packages\tensorflow\python\eager\context.py", line 28, in <module>
    from tensorflow.core.framework import function_pb2
  File "C:\Users\karth\AppData\Roaming\Python\Python37\site-packages\tensorflow\core\framework\function_pb2.py", line 16, in <module>
    from tensorflow.core.framework import attr_value_pb2 as tensorflow_dot_core_dot_framework_dot_attr__value__pb2
  File "C:\Users\karth\AppData\Roaming\Python\Python37\site-packages\tensorflow\core\framework\attr_value_pb2.py", line 16, in <module>
    from tensorflow.core.framework import tensor_pb2 as tensorflow_dot_core_dot_framework_dot_tensor__pb2
  File "C:\Users\karth\AppData\Roaming\Python\Python37\site-packages\tensorflow\core\framework\tensor_pb2.py", line 16, in <module>
    from tensorflow.core.framework import resource_handle_pb2 as tensorflow_dot_core_dot_framework_dot_resource__handle__pb2
  File "C:\Users\karth\AppData\Roaming\Python\Python37\site-packages\tensorflow\core\framework\resource_handle_pb2.py", line 16, in <module>
    from tensorflow.core.framework import tensor_shape_pb2 as tensorflow_dot_core_dot_framework_dot_tensor__shape__pb2
  File "C:\Users\karth\AppData\Roaming\Python\Python37\site-packages\tensorflow\core\framework\tensor_shape_pb2.py", line 42, in <module>
    serialized_options=None, file=DESCRIPTOR),
  File "C:\Users\karth\anaconda3\envs\arxiv_summarizer\lib\site-packages\google\protobuf\descriptor.py", line 561, in __new__
    _message.Message._CheckCalledFromGeneratedFile()
TypeError: Descriptors cannot not be created directly.
If this call came from a _pb2.py file, your generated code is out of date and must be regenerated with protoc >= 3.19.0.
If you cannot immediately regenerate your protos, some other possible workarounds are:
 1. Downgrade the protobuf package to 3.20.x or lower.
 2. Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python (but this will use pure-Python parsing and will be much slower).

More information: https://developers.google.com/protocol-buffers/docs/news/2022-05-06#python-updates
```

<br>

I resolved it by installing **protobuf**. 

```
pip install protobuf==3.6.0

Collecting protobuf==3.6.0
  Using cached protobuf-3.6.0-py2.py3-none-any.whl (390 kB)
Requirement already satisfied: setuptools in c:\users\karth\anaconda3\envs\arxiv_summarizer\lib\site-packages (from protobuf==3.6.0) (65.6.3)
Requirement already satisfied: six>=1.9 in c:\users\karth\anaconda3\envs\arxiv_summarizer\lib\site-packages (from protobuf==3.6.0) (1.15.0)
Installing collected packages: protobuf
  Attempting uninstall: protobuf
    Found existing installation: protobuf 4.23.3
    Uninstalling protobuf-4.23.3:
      Successfully uninstalled protobuf-4.23.3
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
tensorflow-intel 2.11.0 requires keras<2.12,>=2.11.0, but you have keras 2.10.0 which is incompatible.
tensorflow-intel 2.11.0 requires protobuf<3.20,>=3.9.2, but you have protobuf 3.6.0 which is incompatible.
tensorflow-intel 2.11.0 requires tensorboard<2.12,>=2.11, but you have tensorboard 2.0.2 which is incompatible.
tensorflow-intel 2.11.0 requires tensorflow-estimator<2.12,>=2.11.0, but you have tensorflow-estimator 2.0.1 which is incompatible.
tensorflow 2.3.0 requires h5py<2.11.0,>=2.10.0, but you have h5py 3.1.0 which is incompatible.
tensorflow 2.3.0 requires numpy<1.19.0,>=1.16.0, but you have numpy 1.21.6 which is incompatible.
tensorflow 2.3.0 requires protobuf>=3.9.2, but you have protobuf 3.6.0 which is incompatible.
tensorflow 2.3.0 requires tensorboard<3,>=2.3.0, but you have tensorboard 2.0.2 which is incompatible.
tensorflow 2.3.0 requires tensorflow-estimator<2.4.0,>=2.3.0, but you have tensorflow-estimator 2.0.1 which is incompatible.
tensorflow-text 2.10.0 requires tensorflow<2.11,>=2.10.0; platform_machine != "arm64" or platform_system != "Darwin", but you have tensorflow 2.3.0 which is incompatible.      
tensorflow-metadata 1.2.0 requires absl-py<0.13,>=0.9, but you have absl-py 1.4.0 which is incompatible.
tensorflow-metadata 1.2.0 requires protobuf<4,>=3.13, but you have protobuf 3.6.0 which is incompatible.
tensorflow-hub 0.13.0 requires protobuf>=3.19.6, but you have protobuf 3.6.0 which is incompatible.
tensorflow-datasets 2.1.0 requires protobuf>=3.6.1, but you have protobuf 3.6.0 which is incompatible.
googleapis-common-protos 1.56.3 requires protobuf<5.0.0dev,>=3.15.0, but you have protobuf 3.6.0 which is incompatible.
google-api-core 2.8.2 requires protobuf<5.0.0dev,>=3.15.0, but you have protobuf 3.6.0 which is incompatible.
Successfully installed protobuf-3.6.0
```

<br>

Now, tensorflow-text is giving DLL load failed error

```
(arxiv_summarizer) C:\Users\karth\OneDrive\Documents\Other_Resources\ProfileProject\pegasus>python evaluate.py --params=aeslc_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=5,beam_alpha=0.6 --model_dir=ckpt/pegasus_ckpt/aeslc

Traceback (most recent call last):
  File "evaluate.py", line 22, in <module>
    from pegasus.data import infeed
  File "C:\Users\karth\OneDrive\Documents\Other_Resources\ProfileProject\pegasus\pegasus\data\infeed.py", line 18, in <module>
    from pegasus.ops import public_parsing_ops
  File "C:\Users\karth\OneDrive\Documents\Other_Resources\ProfileProject\pegasus\pegasus\ops\public_parsing_ops.py", line 23, in <module>
    import tensorflow_text as tf_text
  File "C:\Users\karth\anaconda3\envs\arxiv_summarizer\lib\site-packages\tensorflow_text\__init__.py", line 20, in <module>
    from tensorflow_text.core.pybinds import tflite_registrar
ImportError: DLL load failed: The specified procedure could not be found.
```

<br>

I upgraded the **tensorflow-text** version, still the **DLL load failed** error persisted. 

```
pip show tensorflow-text

Name: tensorflow-text
Version: 2.10.0
Summary: TF.Text is a TensorFlow library of text related ops, modules, and subgraphs.
Home-page: http://github.com/tensorflow/text
Author: Google Inc.
Author-email: packages@tensorflow.org
License: Apache 2.0
Location: c:\users\karth\anaconda3\envs\arxiv_summarizer\lib\site-packages
Requires: tensorflow, tensorflow-hub
Required-by: 
```

<br>

Changing the **tensorflow-text** version

```
pip install tensorflow_text==2.8.2


Collecting tensorflow_text==2.8.2
  Downloading tensorflow_text-2.8.2-cp37-cp37m-win_amd64.whl (2.5 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.5/2.5 MB 4.4 MB/s eta 0:00:00
Requirement already satisfied: tensorflow-hub>=0.8.0 in c:\users\karth\anaconda3\envs\arxiv_summarizer\lib\site-packages (from tensorflow_text==2.8.2) (0.13.0)
Collecting tensorflow<2.9,>=2.8.0
  Downloading tensorflow-2.8.4-cp37-cp37m-win_amd64.whl (438.3 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 438.3/438.3 MB ? eta 0:00:00
Requirement already satisfied: h5py>=2.9.0 in c:\users\karth\anaconda3\envs\arxiv_summarizer\lib\site-packages (from tensorflow<2.9,>=2.8.0->tensorflow_text==2.8.2) (3.1.0)
Requirement already satisfied: libclang>=9.0.1 in c:\users\karth\anaconda3\envs\arxiv_summarizer\lib\site-packages (from tensorflow<2.9,>=2.8.0->tensorflow_text==2.8.2) (16.0.0)
Collecting keras<2.9,>=2.8.0rc0
  Downloading keras-2.8.0-py2.py3-none-any.whl (1.4 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.4/1.4 MB 3.2 MB/s eta 0:00:00
Requirement already satisfied: grpcio<2.0,>=1.24.3 in c:\users\karth\anaconda3\envs\arxiv_summarizer\lib\site-packages (from tensorflow<2.9,>=2.8.0->tensorflow_text==2.8.2) (1.48.2)
Requirement already satisfied: wrapt>=1.11.0 in c:\users\karth\anaconda3\envs\arxiv_summarizer\lib\site-packages (from tensorflow<2.9,>=2.8.0->tensorflow_text==2.8.2) (1.12.1) 
Requirement already satisfied: opt-einsum>=2.3.2 in c:\users\karth\appdata\roaming\python\python37\site-packages (from tensorflow<2.9,>=2.8.0->tensorflow_text==2.8.2) (3.3.0)  
Requirement already satisfied: keras-preprocessing>=1.1.1 in c:\users\karth\anaconda3\envs\arxiv_summarizer\lib\site-packages (from tensorflow<2.9,>=2.8.0->tensorflow_text==2.8.2) (1.1.2)
Collecting protobuf<3.20,>=3.9.2
  Using cached protobuf-3.19.6-cp37-cp37m-win_amd64.whl (896 kB)
Collecting tensorflow-estimator<2.9,>=2.8
  Downloading tensorflow_estimator-2.8.0-py2.py3-none-any.whl (462 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 462.3/462.3 kB 1.6 MB/s eta 0:00:00
Requirement already satisfied: setuptools in c:\users\karth\anaconda3\envs\arxiv_summarizer\lib\site-packages (from tensorflow<2.9,>=2.8.0->tensorflow_text==2.8.2) (65.6.3)
Requirement already satisfied: google-pasta>=0.1.1 in c:\users\karth\appdata\roaming\python\python37\site-packages (from tensorflow<2.9,>=2.8.0->tensorflow_text==2.8.2) (0.2.0)Requirement already satisfied: astunparse>=1.6.0 in c:\users\karth\appdata\roaming\python\python37\site-packages (from tensorflow<2.9,>=2.8.0->tensorflow_text==2.8.2) (1.6.3)  
Requirement already satisfied: absl-py>=0.4.0 in c:\users\karth\anaconda3\envs\arxiv_summarizer\lib\site-packages (from tensorflow<2.9,>=2.8.0->tensorflow_text==2.8.2) (1.4.0) 
Requirement already satisfied: six>=1.12.0 in c:\users\karth\anaconda3\envs\arxiv_summarizer\lib\site-packages (from tensorflow<2.9,>=2.8.0->tensorflow_text==2.8.2) (1.15.0)   
Requirement already satisfied: gast>=0.2.1 in c:\users\karth\anaconda3\envs\arxiv_summarizer\lib\site-packages (from tensorflow<2.9,>=2.8.0->tensorflow_text==2.8.2) (0.3.3)
Requirement already satisfied: typing-extensions>=3.6.6 in c:\users\karth\anaconda3\envs\arxiv_summarizer\lib\site-packages (from tensorflow<2.9,>=2.8.0->tensorflow_text==2.8.2) (3.7.4.3)
Requirement already satisfied: termcolor>=1.1.0 in c:\users\karth\anaconda3\envs\arxiv_summarizer\lib\site-packages (from tensorflow<2.9,>=2.8.0->tensorflow_text==2.8.2) (1.1.0)
Requirement already satisfied: numpy>=1.20 in c:\users\karth\anaconda3\envs\arxiv_summarizer\lib\site-packages (from tensorflow<2.9,>=2.8.0->tensorflow_text==2.8.2) (1.21.6)
Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in c:\users\karth\anaconda3\envs\arxiv_summarizer\lib\site-packages (from tensorflow<2.9,>=2.8.0->tensorflow_text==2.8.2) (0.31.0)
Requirement already satisfied: flatbuffers>=1.12 in c:\users\karth\anaconda3\envs\arxiv_summarizer\lib\site-packages (from tensorflow<2.9,>=2.8.0->tensorflow_text==2.8.2) (23.5.26)
Collecting tensorboard<2.9,>=2.8
  Downloading tensorboard-2.8.0-py3-none-any.whl (5.8 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5.8/5.8 MB 5.7 MB/s eta 0:00:00
Requirement already satisfied: wheel<1.0,>=0.23.0 in c:\users\karth\anaconda3\envs\arxiv_summarizer\lib\site-packages (from astunparse>=1.6.0->tensorflow<2.9,>=2.8.0->tensorflow_text==2.8.2) (0.37.1)
Requirement already satisfied: cached-property in c:\users\karth\anaconda3\envs\arxiv_summarizer\lib\site-packages (from h5py>=2.9.0->tensorflow<2.9,>=2.8.0->tensorflow_text==2.8.2) (1.5.2)
Requirement already satisfied: tensorboard-data-server<0.7.0,>=0.6.0 in c:\users\karth\anaconda3\envs\arxiv_summarizer\lib\site-packages (from tensorboard<2.9,>=2.8->tensorflow<2.9,>=2.8.0->tensorflow_text==2.8.2) (0.6.1)
Requirement already satisfied: google-auth<3,>=1.6.3 in c:\users\karth\anaconda3\envs\arxiv_summarizer\lib\site-packages (from tensorboard<2.9,>=2.8->tensorflow<2.9,>=2.8.0->tensorflow_text==2.8.2) (1.35.0)
Requirement already satisfied: werkzeug>=0.11.15 in c:\users\karth\appdata\roaming\python\python37\site-packages (from tensorboard<2.9,>=2.8->tensorflow<2.9,>=2.8.0->tensorflow_text==2.8.2) (2.2.3)
Requirement already satisfied: google-auth-oauthlib<0.5,>=0.4.1 in c:\users\karth\appdata\roaming\python\python37\site-packages (from tensorboard<2.9,>=2.8->tensorflow<2.9,>=2.8.0->tensorflow_text==2.8.2) (0.4.6)
Requirement already satisfied: requests<3,>=2.21.0 in c:\users\karth\appdata\roaming\python\python37\site-packages (from tensorboard<2.9,>=2.8->tensorflow<2.9,>=2.8.0->tensorflow_text==2.8.2) (2.28.2)
Requirement already satisfied: tensorboard-plugin-wit>=1.6.0 in c:\users\karth\anaconda3\envs\arxiv_summarizer\lib\site-packages (from tensorboard<2.9,>=2.8->tensorflow<2.9,>=2.8.0->tensorflow_text==2.8.2) (1.8.1)
Requirement already satisfied: markdown>=2.6.8 in c:\users\karth\appdata\roaming\python\python37\site-packages (from tensorboard<2.9,>=2.8->tensorflow<2.9,>=2.8.0->tensorflow_text==2.8.2) (3.4.3)
Requirement already satisfied: rsa<5,>=3.1.4 in c:\users\karth\anaconda3\envs\arxiv_summarizer\lib\site-packages (from google-auth<3,>=1.6.3->tensorboard<2.9,>=2.8->tensorflow<2.9,>=2.8.0->tensorflow_text==2.8.2) (4.9)
Requirement already satisfied: pyasn1-modules>=0.2.1 in c:\users\karth\anaconda3\envs\arxiv_summarizer\lib\site-packages (from google-auth<3,>=1.6.3->tensorboard<2.9,>=2.8->tensorflow<2.9,>=2.8.0->tensorflow_text==2.8.2) (0.3.0)
Requirement already satisfied: cachetools<5.0,>=2.0.0 in c:\users\karth\anaconda3\envs\arxiv_summarizer\lib\site-packages (from google-auth<3,>=1.6.3->tensorboard<2.9,>=2.8->tensorflow<2.9,>=2.8.0->tensorflow_text==2.8.2) (4.2.4)
Requirement already satisfied: requests-oauthlib>=0.7.0 in c:\users\karth\appdata\roaming\python\python37\site-packages (from google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.9,>=2.8->tensorflow<2.9,>=2.8.0->tensorflow_text==2.8.2) (1.3.1)
Requirement already satisfied: importlib-metadata>=4.4 in c:\users\karth\anaconda3\envs\arxiv_summarizer\lib\site-packages (from markdown>=2.6.8->tensorboard<2.9,>=2.8->tensorflow<2.9,>=2.8.0->tensorflow_text==2.8.2) (4.8.3)
Requirement already satisfied: charset-normalizer<4,>=2 in c:\users\karth\appdata\roaming\python\python37\site-packages (from requests<3,>=2.21.0->tensorboard<2.9,>=2.8->tensorflow<2.9,>=2.8.0->tensorflow_text==2.8.2) (3.1.0)
Requirement already satisfied: urllib3<1.27,>=1.21.1 in c:\users\karth\anaconda3\envs\arxiv_summarizer\lib\site-packages (from requests<3,>=2.21.0->tensorboard<2.9,>=2.8->tensorflow<2.9,>=2.8.0->tensorflow_text==2.8.2) (1.26.16)
Requirement already satisfied: certifi>=2017.4.17 in c:\users\karth\anaconda3\envs\arxiv_summarizer\lib\site-packages (from requests<3,>=2.21.0->tensorboard<2.9,>=2.8->tensorflow<2.9,>=2.8.0->tensorflow_text==2.8.2) (2022.12.7)
Requirement already satisfied: idna<4,>=2.5 in c:\users\karth\appdata\roaming\python\python37\site-packages (from requests<3,>=2.21.0->tensorboard<2.9,>=2.8->tensorflow<2.9,>=2.8.0->tensorflow_text==2.8.2) (3.4)
Requirement already satisfied: MarkupSafe>=2.1.1 in c:\users\karth\appdata\roaming\python\python37\site-packages (from werkzeug>=0.11.15->tensorboard<2.9,>=2.8->tensorflow<2.9,>=2.8.0->tensorflow_text==2.8.2) (2.1.2)
Requirement already satisfied: zipp>=0.5 in c:\users\karth\anaconda3\envs\arxiv_summarizer\lib\site-packages (from importlib-metadata>=4.4->markdown>=2.6.8->tensorboard<2.9,>=2.8->tensorflow<2.9,>=2.8.0->tensorflow_text==2.8.2) (3.6.0)
Requirement already satisfied: pyasn1<0.6.0,>=0.4.6 in c:\users\karth\anaconda3\envs\arxiv_summarizer\lib\site-packages (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard<2.9,>=2.8->tensorflow<2.9,>=2.8.0->tensorflow_text==2.8.2) (0.5.0)
Requirement already satisfied: oauthlib>=3.0.0 in c:\users\karth\appdata\roaming\python\python37\site-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.9,>=2.8->tensorflow<2.9,>=2.8.0->tensorflow_text==2.8.2) (3.2.2)
Installing collected packages: tensorflow-estimator, keras, protobuf, tensorboard, tensorflow, tensorflow_text
  Attempting uninstall: tensorflow-estimator
    Found existing installation: tensorflow-estimator 2.0.1
    Uninstalling tensorflow-estimator-2.0.1:
      Successfully uninstalled tensorflow-estimator-2.0.1
  Attempting uninstall: keras
    Found existing installation: keras 2.10.0
    Uninstalling keras-2.10.0:
      Successfully uninstalled keras-2.10.0
  Attempting uninstall: protobuf
    Found existing installation: protobuf 3.6.0
    Uninstalling protobuf-3.6.0:
      Successfully uninstalled protobuf-3.6.0
  Attempting uninstall: tensorboard
    Found existing installation: tensorboard 2.0.2
    Uninstalling tensorboard-2.0.2:
      Successfully uninstalled tensorboard-2.0.2
  Attempting uninstall: tensorflow
    Found existing installation: tensorflow 2.3.0
    Uninstalling tensorflow-2.3.0:
      Successfully uninstalled tensorflow-2.3.0
  Attempting uninstall: tensorflow_text
    Found existing installation: tensorflow-text 2.10.0
    Uninstalling tensorflow-text-2.10.0:
      Successfully uninstalled tensorflow-text-2.10.0
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
tensorflow-intel 2.11.0 requires keras<2.12,>=2.11.0, but you have keras 2.8.0 which is incompatible.
tensorflow-intel 2.11.0 requires tensorboard<2.12,>=2.11, but you have tensorboard 2.8.0 which is incompatible.
tensorflow-intel 2.11.0 requires tensorflow-estimator<2.12,>=2.11.0, but you have tensorflow-estimator 2.8.0 which is incompatible.
tensorflow-metadata 1.2.0 requires absl-py<0.13,>=0.9, but you have absl-py 1.4.0 which is incompatible.
Successfully installed keras-2.8.0 protobuf-3.19.6 tensorboard-2.8.0 tensorflow-2.8.4 tensorflow-estimator-2.8.0 tensorflow_text-2.8.2
```

<br>

The error persisted

```
(arxiv_summarizer) C:\Users\karth\OneDrive\Documents\Other_Resources\ProfileProject\pegasus>python evaluate.py --params=aeslc_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=5,beam_alpha=0.6 --model_dir=ckpt/pegasus_ckpt/aeslc

Traceback (most recent call last):
  File "evaluate.py", line 22, in <module>
    from pegasus.data import infeed
  File "C:\Users\karth\OneDrive\Documents\Other_Resources\ProfileProject\pegasus\pegasus\data\infeed.py", line 18, in <module>
    from pegasus.ops import public_parsing_ops
  File "C:\Users\karth\OneDrive\Documents\Other_Resources\ProfileProject\pegasus\pegasus\ops\public_parsing_ops.py", line 23, in <module>
    import tensorflow_text as tf_text
  File "C:\Users\karth\anaconda3\envs\arxiv_summarizer\lib\site-packages\tensorflow_text\__init__.py", line 21, in <module>
    from tensorflow_text.python import keras
  File "C:\Users\karth\anaconda3\envs\arxiv_summarizer\lib\site-packages\tensorflow_text\python\keras\__init__.py", line 21, in <module>
    from tensorflow_text.python.keras.layers import *
  File "C:\Users\karth\anaconda3\envs\arxiv_summarizer\lib\site-packages\tensorflow_text\python\keras\layers\__init__.py", line 22, in <module>
    from tensorflow_text.python.keras.layers.tokenization_layers import *
  File "C:\Users\karth\anaconda3\envs\arxiv_summarizer\lib\site-packages\tensorflow_text\python\keras\layers\tokenization_layers.py", line 25, in <module>
    from tensorflow_text.python.ops import unicode_script_tokenizer
  File "C:\Users\karth\anaconda3\envs\arxiv_summarizer\lib\site-packages\tensorflow_text\python\ops\__init__.py", line 23, in <module>
    from tensorflow_text.core.pybinds.pywrap_fast_wordpiece_tokenizer_model_builder import build_fast_wordpiece_model
ImportError: DLL load failed: The specified procedure could not be found.
```

<br>

As a conclusion, the error with **tensorflow-text** persisted, this was a blocker to evaluate pegasus model. 

<br>

This blog is a documentation of my process to use pegasus model. Even though huggingface provides an easy interface to use pegasus model, I wanted to install it from scratch to gain more control over the predicition steps. Due to technical challenges, I am moving to my next project. If anyone finds a solution to fix **tensorflow-text**, please feel free to contact me. 

