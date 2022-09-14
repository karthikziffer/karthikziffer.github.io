---
layout: post
title: "tflite inference in raspberry pi"
author: "Karthik"
categories: journal
tags: [documentation,sample]




---



<br>

## Raspberry pi OS

```

pi@raspberrypi:~ $ cat /etc/os-release

PRETTY_NAME="Raspbian GNU/Linux 11 (bullseye)"
NAME="Raspbian GNU/Linux"
VERSION_ID="11"
VERSION="11 (bullseye)"
VERSION_CODENAME=bullseye
ID=raspbian
ID_LIKE=debian
HOME_URL="http://www.raspbian.org/"
SUPPORT_URL="http://www.raspbian.org/RaspbianForums"
BUG_REPORT_URL="http://www.raspbian.org/RaspbianBugs"

```

<br>

## Architecture

```
pi@raspberrypi:~ $ uname -m

armv7l
```



<br>

## Install Tensorflow lite



```
sudo apt update
sudo apt upgrade -y
```

```
echo "deb [signed-by=/usr/share/keyrings/coral-edgetpu-archive-keyring.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
```

```
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo tee /usr/share/keyrings/coral-edgetpu-archive-keyring.gpg >/dev/null
```

```
sudo apt update
```

```
sudo apt install python3-tflite-runtime libatlas-base-dev
```

```
python3
```

```
from tflite_runtime.interpreter import Interpreter
```



<br>



## Tensorflow lite inference script

```
interpreter = Interpreter(model_path="converted_model.tflite")
interpreter.allocate_tensors()
```

```
# Get input and output tensors.

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
```

```
# Generating sample input data of shape (1,1,2800)

import numpy as np
input_data = np.array( np.random.random((1,1,2800)), dtype=np.float32)

```

```
 interpreter.set_tensor(input_details[0]['index'], input_data)
```

```
interpreter.invoke()
```

```
output_data = interpreter.get_tensor(output_details[0]['index'])
```



output_data is the model output