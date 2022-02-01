---
layout: post
title: "Gradio Tutorial"
author: "Karthik"
categories: journal
tags: [documentation,sample]




---



I recently had a POC to present to my team. It was an image similarity project in the retail space. I was looking for a simple tool to demo the top N similar image for a given query image. I did not want to spend much time in learning the tool, but I started with [Streamlit](https://streamlit.io/), since it was famous than [Gradio](https://gradio.app/). [Streamlit](https://streamlit.io/) is a good package, which provides controlled options to build but requires steep learning curve to make necessary changes. I chose [Gradio](https://gradio.app/) for my image similarity demo over Streamlit because it was simple to implement for my use case. On the other hand, I faced some issues because I was using it in my office laptop, so be prepared. It worked flawlessly in my personal laptop. 



My use case is as follows:

1. Take an input image
2. The image needs different augmentations
3. Returns five most similar images. 

 

Installation:

```
 pip install gradio
```



Let's create a simple application that takes input image and returns 90 degree rotated image



![Capture1.png](https://i.postimg.cc/Hx2LjTQ6/Capture1.png)

The problem with above code is, whenever you execute this code, it runs on a new port. To avoid it, we must close all the active port connections before starting. 

```
import gradio as gr
import numpy as np
import cv2

# close all open ports
gr.close_all()

def output_image_operation(input_img):
    rotated_img = cv2.rotate(input_img, cv2.cv2.ROTATE_90_CLOCKWISE)
    return rotated_img         
  

iface = gr.Interface(output_image_operation,
                     gr.inputs.Image(shape=(200, 200)), "image")

iface.launch()
```





Now, let's output two images



```
import gradio as gr
import numpy as np
import cv2

# close all open ports
gr.close_all()

def output_image_operation(input_img):
    rotated_img = cv2.rotate(input_img, cv2.cv2.ROTATE_90_CLOCKWISE)
    gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY )
    return rotated_img, gray_img        
  

iface = gr.Interface(output_image_operation,
                     gr.inputs.Image(shape=(200, 200)), 
                     ["image", "image"])

iface.launch()
```

![Capture2.png](https://i.postimg.cc/CKK0JbDs/Capture2.png)



Now, let's add some text below the images

```
import gradio as gr
import numpy as np
import cv2

# close all open ports
gr.close_all()

def output_image_operation(input_img):
    rotated_img = cv2.rotate(input_img, cv2.cv2.ROTATE_90_CLOCKWISE)
    gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY )
    return rotated_img, 'Rotated Image', gray_img, 'Grayscale Image'        
  

iface = gr.Interface(output_image_operation,
                     gr.inputs.Image(shape=(200, 200)), 
                     ["image", "text", "image", "text"])

iface.launch()
```





![Capture3.png](https://i.postimg.cc/jdrrmpP5/Capture3.png)

The text is given has **OUTPUT2** and **OUTPUT3**, but I could not find any alternative way to display associated text information. In my use case, I displayed the cosine similarity score in the text field, hence the **OUTPUT2** and **OUTPUT4** labels were ignored. 

<br>

Click on the pencil icon to edit the input image



![Capture4.png](https://i.postimg.cc/3J72Nw2H/Capture4.png)





<br>



That's it for the tutorial. Finally, congratulations to the Gradio Team for joining Hugging Face Team :)