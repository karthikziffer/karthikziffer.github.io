---
layout: post
title: "All about Seq2seq"
author: "Karthik"
categories: journal
tags: [documentation,sample]
---





<br>

<br>

<br>

---





<br>



## [Sequence to Sequence Learning](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)




Sequence to sequence model comprises of two sequence networks one at the encoder side and other at the decoder side. These sequence networks are RNN. The encoder networks maps the input sequence to a vector of a fixed dimension. The decoder network decodes the target sequence from the encoded vector.

Seq2seq networks find application in Machine Translation , Speech Recognition , Time Series etc.



>A useful property of the LSTM is that it learns to map an input sentence of variable length into
>a fixed-dimensional vector representation. Given that translations tend to be paraphrases of the
>source sentences, the translation objective encourages the LSTM to find sentence representations
>that capture their meaning, as sentences with similar meanings are close to each other while different sentences meanings will be far. A qualitative evaluation supports this claim, showing that our model
>is aware of word order and is fairly invariant to the active and passive voice.



> Surprisingly, the LSTM did not suffer on very long sentences, despite the recent experience of other
> researchers with related architectures. We were able to do well on long sentences because we
> reversed the order of words in the source sentence but not the target sentences in the training and test set. By doing so, we introduced many short term dependencies that made the optimization problem much simpler . As a result, SGD could learn LSTMs that had no trouble with
> long sentences. The simple trick of reversing the words in the source sentence is one of the key
> technical contributions of this work.



<br>

### The Model



![Model](https://www.lucidchart.com/publicSegments/view/01a871b6-4d49-4b73-929d-1dbd57de301c/image.png)

<br>

The input sequence to the RNN which computes the output sequence. RNN maps the input sequence to the output sequence. There are two scenarios, when sequence are of fixed sequence length and variable length. Variable length sequences are not addressed in this paper. 

>However, it is not clear how to apply an RNN to problems whose input and the output sequences have different lengths with complicated and non-monotonic relationships.



Input Sequence can be a sequence data vector. The input english sentence incase of English to French Machine translation application. 

<br>
$$
(x_1 ,x_2 , ...x_n)
$$
Output Sequence can be the translated french sentence 

<br>
$$
(y_1 , y_2 ,...y_n)
$$

The hidden states in the RNN are computed by the sigmoid activation on the weighted sum of current input and previous hidden state.

<br>
$$
h_t = \sigma (W^{hx}.x_t + W^{hh}.h_{t-1})
$$

$$
y_t = W^{yh}.h_t
$$





<br>

Since the input sentence can be long, it is challenging to map longer dependencies present in the input sentence. The depencies can be singluar/plural grammatical behaviour present in the input sentence. 

- **Attention Networks** perform well in capturing long sentence dependencies in the sentence. 
- LSTM has **forget gate** and **output gate** which captures this long term dependency information. 

>It would be difficult to train the RNNs due to the resulting long term dependencies. However, the Long Short-Term Memory (LSTM) is known to learn problems with long range temporal dependencies, so an LSTM may succeed in this setting.



The goal of LSTM is to estimate the conditional probability :

<br>
$$
P(y_1....y_{T'} | x_1 ....x_T)
$$
Input sequence length is **T** and output sequence length is **T'** . The input sequence length may differ from that of output sequence.



>The goal of the LSTM is to estimate the conditional probability p(y1, . . . , yT′ |x1, . . . , xT ) where
>(x1, . . . , xT ) is an input sequence and y1, . . . , yT′ is its corresponding output sequence whose length
>T′ may differ from T.



The encoder network takes the input sequence and returns a fixed dimensional vector representations **v**.  The decoder network estimates the conditional probability. The decoder's first cell hidden state is set to the representation vector **v**. 



>The LSTM computes this conditional probability by first obtaining the fixed dimensional
>representation **v** of the input sequence (x1, . . . , xT ) given by the last hidden state of the
>LSTM, and then computing the probability of y1, . . . , yT′ with a standard LSTM-LM formulation
>whose initial hidden state is set to the representation v of x1, . . . , xT.

<br>

$$
P(y_1, ... , y_{T'}) =  \  \Pi^{T'}_{t =1} \ \  P(y_t | v , y_1 , ....,y_{t-1})
$$

The below mentioned Probablity distribution is represented with a softmax over all words in the vocabulary. 

<br>
$$
P(y_t | v , y_1 , ....,y_{t-1})
$$


>we require that each sentence ends with a special end-of-sentence symbol **<EOS>**, which enables the model to define a distribution over sequences of all possible lengths. 



<br>

### Decoding and Rescoring



The LSTM is trained on many sentences. We train the LSTM to provide maximum of correct translation (**T**) log probability from the given source sentence (**S**).

The training objective is given by,

<br>

$$
1/|S| . \sum_{(T,S)  \epsilon  \delta }  \log{p(T|S)}
$$

$$
^\star \ \delta \ is \ the \ training \ set.
$$

Once training is complete , we produce translations by finding the most likely translation according to the LSTM.

<br>

$$
\widehat{T} = arg \ max_T \ p(T|S)
$$




#### Beam Search

---





>We search for the most likely translation using a simple left-to-right **beam search decoder** which
>maintains a small number **B** of partial hypotheses, where a partial hypothesis is a prefix of some
>translation.

 

>At each timestep we extend each partial hypothesis in the beam with every possible word in the vocabulary. This greatly increases the number of the hypotheses so we discard all but the **B** most likely hypotheses according to the model’s log probability.



>As soon as the **<EOS>** symbol is appended to a hypothesis, it is removed from the beam and is added to the set of complete hypotheses.





---



### [STATE-OF-THE-ART SPEECH RECOGNITION WITH SEQUENCE-TO-SEQUENCE MODELS](https://sci-hub.tw/https://ieeexplore.ieee.org/abstract/document/8462105)





In the paper, Sequence to sequence models are used for Speech Recognition Task.  An Attention based encoder-decoder architecture with LAS (Listen , Attend , Spell) capability in single neural architecture. LAS features can be further employed for decoding acoustic , pronunciation , language model components.

<br>

#### Model Architecture

![Model Architecture](https://www.lucidchart.com/publicSegments/view/3194636f-f84a-4427-9eeb-13982ffdaa4a/image.png)



***The model comprises of three modules:***

- Encoder
- Attender (Attention Module)
- Decoder



The Encoder maps the input sequence to a high level feature representations (h enc).  This representations are fed to the attention module , which determines which encoder features in **h enc** should be attended to in order to predict the next output **yi**. Finally the output of the attention module is passed to the decoder which takes the attention context **Ci** as well as the embedding of the previous prediction **yi-1** .

 The decoder produces a probability distribution.

<br>
$$
P(y_i | y_{i-1} , ....., y_0 , x)
$$











---





### Other Similar Applications:



1. [GENERALIZATION WITHOUT SYSTEMATICITY: ON THE COMPOSITIONAL SKILLS OF SEQUENCE -TO-SEQUENCE RECURRENT NETWORKS](http://proceedings.mlr.press/v80/lake18a/lake18a.pdf)
2. [SCALABLE SENTIMENT FOR SEQUENCE-TO-SEQUENCE CHATBOT RESPONSE WITH PERFORMANCE ANALYSIS](https://arxiv.org/pdf/1804.02504.pdf)
3. [MULTI-DIALECT SPEECH RECOGNITION WITH A SINGLE SEQUENCE-TO-SEQUENCE MODEL](https://arxiv.org/pdf/1712.01541.pdf)

