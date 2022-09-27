---
layout: post
title: "Convert PyTorch to TFLite"
author: "Karthik"
categories: journal
tags: [documentation,sample]





---



<br>

## PyTorch LSTM model architecture

<br>



```
LSTM1(
  (lstm1): LSTM(2800, 5, batch_first=True)
  (linear): Linear(in_features=5, out_features=5, bias=True)
  (softmax): Softmax(dim=None)
)
```





<br>

```
class LSTM1(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm1 = torch.nn.LSTM(input_size, hidden_size,  batch_first=True, bidirectional=False)
        self.linear = torch.nn.Linear(hidden_size, output_size)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        output, (hn, cn) = self.lstm1(x)
        x = self.linear(output)
        return x
```



<br>

## Conversion

<br>

```
pip install tf-nightly==2.11.0.dev20220927
```

<br>

```
pip show tensorflow==2.8.2+zzzcolab20220719082949
```

<br>

```
pip show torch==1.12.1+cu113
```

<br>

```
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers, optimizers
print(tf.__version__)
print("Eagerly enabled: ", tf.executing_eagerly())
```



<br>

```
MODEL_PATH = "/content/gdrive/MyDrive/ResearchProject/NewData/10_hidden_layer_5_aug30__0.96432_multiclass_model.pt"

input_size = 2800 #number of features
hidden_size = 5 #number of features in hidden state
num_classes = 5 #number of output classes 
lstm1 = LSTM1(input_size, hidden_size, num_classes)

lstm1 = torch.load(MODEL_PATH, map_location='cpu')
lstm1.eval()
```

<br>

```
# torch model output
print(X_train.shape)
torch.Size([37500, 1, 2800])

sample_input = X_train[0:1]
print(sample_input.shape)
torch.Size([1, 1, 2800])

print(X_train.shape, sample_input.shape, type(sample_input), type(X_train))
torch.Size([37500, 1, 2800]) torch.Size([1, 1, 2800]) <class 'torch.Tensor'> <class 'torch.Tensor'>

torch_out = lstm1(sample_input)
print(torch_out)
tensor([[[-0.0377,  0.0866, -0.0598,  1.0364, -0.0254]]], grad_fn=<ViewBackward0>)
      
      
torch.onnx.export(model = lstm1, args = sample_input, f = "./LSTM.onnx", opset_version=12, input_names=['input'], output_names=['output'] 				
)
```



<br>

```
pip install onnx==1.12.0
pip install onnxruntime==1.12.1
pip install onnx_tf==1.6
pip install tensorflow-addons==0.11.2
```

<br>

```
import onnx
from onnx_tf.backend import prepare

onnx_model = onnx.load("./LSTM.onnx")
tf_rep = prepare(onnx_model)
tf_rep.export_graph("./LSTM_tf_2.pb")
```

<br>

```

#TensorFlow freezegraph .pb model file 
converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph('/content/LSTM_tf_2.pb', input_shapes = {'input':[1,1,2800]}, input_arrays = ['input'],output_arrays = ['output'])


converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter.allow_custom_ops=True
converter.experimental_new_converter =True
tflite_model = converter.convert()
```



<br>

```
open("converted_model.tflite", "wb").write(tflite_model)
interpreter = tf.lite.Interpreter(model_path="converted_model.tflite")
interpreter.allocate_tensors()
```

<br>

```
# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
input_shape = input_details[0]['shape']

print(input_shape)
[   1,    1, 2800]
```

<br>

```
interpreter.set_tensor(input_details[0]['index'], np.array( np.random.random((1,1,2800)), dtype=np.float32))

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])

```



