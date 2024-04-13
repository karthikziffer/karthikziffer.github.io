---
layout: post
title: "Dockerize whisper model"
author: "Karthik"
categories: journal
tags: [documentation,sample]
---



# Introduction
What is a whisper model?
Whisper is an open source speech recognition model trained by OpenAI. It enables transcription in multiple languages, as well as translation from those languages into English. 

Because Whisper was trained on a large and diverse dataset and was not fine-tuned to any specific one, it does not beat models that specialize in LibriSpeech performance, a famously competitive benchmark in speech recognition.

About a third of Whisper’s audio dataset is non-English, and it is alternately given the task of transcribing in the original language or translating to English. 

# How to train a whisper model?
There is a repository called whisperX that provides 70X fast automatic speech recognition with additional features such as forced alignment, voice activity detection and speaker diarization.  

# How to install whisperX in local?

conda create --name whisperx python=3.10

conda activate whisperx

conda install pytorch==2.0.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia

pip install git+https://github.com/m-bain/whisperx.git

# Problem

In local system this installation step works perfectly, but when the same steps are tried as a docker image, the dependency of whisperX on [faster_whisper](https://github.com/SYSTRAN/faster-whisper) throws an error and breaks the installation.  



# Solution
#### requirements.txt

```
faster-whisper==0.10.0
torch==2.0.0 
torchvision==0.15.1 
torchaudio==2.0.1
uvicorn==0.27.1
pydantic==2.6.0
Levenshtein==0.24.0
fastapi==0.109.2
pandas==2.0.3
transformers==4.32.1
setuptools>=65
soundfile==0.12.1
nltk
```

#### Dockerfile

```
FROM continuumio/anaconda3

WORKDIR /usr/src/app

RUN apt-get update && apt-get --yes install libsndfile1 && apt-get --yes install  ffmpeg

RUN conda create --name whisperx

ENV PATH /opt/conda/envs/whisperx/bin:$PATH

SHELL ["conda", "run", "-n", "whisperx", "/bin/bash", "-c"]

RUN conda install python=3.10

RUN conda install pip

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install git+https://github.com/m-bain/whisperx.git
```



# Verification

Run this code snippet to verify the successful installation of whisperX

```
import whisperx

device = "cuda" 
audio_file = "audio.mp3"
batch_size = 16 
compute_type = "float16" 


model = whisperx.load_model("large-v2", device, compute_type=compute_type)


model_dir = "/path/"
model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir)

audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size)
print(result["segments"])

```


# Summary
In this blog post, I have shared the dockerfile to dockerize the whisper model inference by overcoming the issue of installing faster_whisper.  The solution is to install faster_whisper seperately as a pip install command. Since the faster_whisper dependency is met, then whisperX smoothly continues the installation steps. 