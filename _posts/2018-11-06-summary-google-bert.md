---
layout: post
title: "Summary - Google BERT"
author: "Karthik"
categories: journal
tags: [documentation,sample]
---




[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)
- Jacob Devlin 
- Ming-Wei Chang 
- Kenton Lee 
- Kristina Toutanova

---
### Abstract

What's BERT
> BERT stands for Bidirectional Encoder Representations from Transformers.

How BERT is unique 
> BERT is designed to pre-train deep bidirectional representations by jointly conditioning on both left and right context in all layers.

How BERT can be used
> As a result, the pre-trained BERT representations can be fine-tuned with just one additional output layer to create state-of-the art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.

BERT's results
> BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE benchmark to 80.4% (7.6% absolute improvement), MultiNLI accuracy to 86.7% (5.6% absolute improvement) and the SQuAD v1.1 question answering Test F1 to 93.2 (1.5 absolute improvement), outperforming human performance by 2.0.

### Introduction

![](https://pli.io/2MkmPq.png)

Feature Based Approach:
> The feature-based approach, such as ELMo (Peters et al., 2018), uses tasks-specific architectures that include the pre-trained representations as additional features.

Fine Tuning Approach:

##### From [OpenAI GPT Blog](https://blog.openai.com/language-unsupervised/)
>> Our system works in two stages; first we train a **[transformer model](https://arxiv.org/pdf/1706.03762.pdf)** on a very large amount of data in an unsupervised manner — using language modeling as a training signal — then we fine-tune this model on much smaller supervised datasets to help it solve specific tasks.

>The fine-tuning approach, such as the Generative Pre-trained Transformer (OpenAI GPT) (Radford et al., 2018), introduces minimal task-specific parameters, and is trained on the downstream tasks by simply fine-tuning the pretrained parameters.

Which is the best approach
>In previous work, both approaches share the same objective function during pre-training, where they use unidirectional language models to learn general language representations.

 What are the limitations in these approaches
 > We argue that current techniques severely restrict the power of the pre-trained representations, especially for the fine-tuning approaches. The major limitation is that standard language models are unidirectional, and this limits the choice of architectures that can be used during pre-training.

Example:
>For example, in OpenAI GPT, the authors use a leftto-right architecture, where every token can only attended to previous tokens in the self-attention layers of the Transformer (Vaswani et al., 2017). Such restrictions are sub-optimal for sentencelevel tasks, and could be devastating when applying fine-tuning based approaches to token-level tasks such as SQuAD question answering (Rajpurkar et al., 2016), where it is crucial to incorporate context from both directions.


What is BERT's speciality

>In this paper, we improve the fine-tuning based approaches by proposing BERT: Bidirectional Encoder Representations from Transformers. BERT addresses the previously mentioned unidirectional constraints by proposing a new pre-training objective: 

![BERT pre-trained objective](https://pli.io/2Mk6WH.png)


>> the “masked language model” (MLM), inspired by the Cloze task (Taylor, 1953). The masked language model randomly masks some of the tokens from the input, and the objective is to predict the original vocabulary id of the masked word based only on its context. Unlike left-to-right language model pre-training, the MLM objective allows the representation to fuse the left and the right context, which allows us to pre-train a deep bidirectional Transformer.

>> We also introduce a “next sentence prediction” task that jointly pre-trains text-pair representations.

BERT's Contribution:

![BERT's Contribution](https://pli.io/2MkDVJ.png)

![Comparision of BERT , OpenAI GPT , ELMo](https://pli.io/2MkbnZ.png)

## BERT Architecture

### Model Architecture.

>In this work, we denote the number of layers (i.e., Transformer blocks) as L, the hidden size as H, and the number of self-attention heads as A. In all cases we set the feed-forward/filter size to be 4H, i.e., 3072 for the H = 768 and 4096 for the H = 1024. We primarily report results on two model sizes:

L - Number of Layers
H - Hidden size
A - Number of self-attention heads

---
> BERTBASE was chosen to have an identical model size as OpenAI GPT for comparison purposes. Critically, however, the BERT Transformer uses bidirectional self-attention, while the GPT Transformer uses constrained self-attention where every token can only attend to context to its left.

 $$ BERT_{BASE} : L=12, H=768, A=12, \\ Total Parameters=110M$$


$$ BERT_{LARGE}: L=24, H=1024, A=16,\\  Total Parameters=340M$$

---
>Transformer is often referred to as a “Transformer encoder” while the left-context-only version is referred to as a “Transformer decoder” since it can be used for text generation.

### Input Representation.

>Our input representation is able to unambiguously represent both a single text sentence or a pair of text sentences (e.g., [Question, Answer]) in one token sequence. For a given token, its input representation is constructed by summing the corresponding token, segment and position embeddings. A visual representation is given below.

![Input representation](https://pli.io/2X0aoQ.png)

The Specifics:
- We use WordPiece embeddings (Wu et al., 2016) with a 30,000 token vocabulary. We denote split word pieces with ##.
- We use learned positional embeddings with supported sequence lengths up to 512 tokens.
- The first token of every sequence is always the special classification embedding ([CLS]). The final hidden state (i.e., output of Transformer) corresponding to this token is used as the aggregate sequence representation for classification tasks. For nonclassification tasks, this vector is ignored.
- Sentence pairs are packed together into a single sequence. We differentiate the sentences in two ways. First, we separate them with a special token ([SEP]). Second, we add a learned sentence A embedding to every token of the first sentence and a sentence B embedding to every token of the second sentence
- For single-sentence inputs we only use the sentence A embeddings

### Pre-training Tasks.

>Unlike Peters et al. (2018) and Radford et al. (2018), we do not use traditional left-to-right or right-to-left language models to pre-train BERT. Instead, we pre-train BERT using two novel unsupervised prediction tasks, described in this section.

![BERT pre-trained objective](https://pli.io/2Mk6WH.png)

+    **Masked Language Model** 

> Intuitively, it is reasonable to believe that a deep bidirectional model is strictly more powerful than either a left-to-right model or the shallow concatenation of a left-to-right and right-toleft model. Unfortunately, standard conditional language models can only be trained left-to-right or right-to-left, since bidirectional conditioning would allow each word to indirectly “see itself” in a multi-layered context.

> In order to train a deep bidirectional representation, we take a straightforward approach of masking some percentage of the input tokens at random, and then predicting only those masked tokens. We refer to this procedure as a “masked LM” (MLM), although it is often referred to as a Cloze task in the literature.

> . In this case, the final hidden vectors corresponding to the mask tokens are fed into an output softmax over the vocabulary, as in a standard LM. In all of our experiments, we mask 15% of all WordPiece tokens in each sequence at random. In contrast to denoising auto-encoders (Vincent et al., 2008), we only predict the masked words rather than reconstructing the entire input.

+ **Next Sentence Prediction**

> Many important downstream tasks such as Question Answering (QA) and Natural Language Inference (NLI) are based on understanding the relationship between two text sentences, which is not directly captured by language modeling. In order to train a model that understands sentence relationships, we pre-train a binarized next sentence prediction task that can be trivially generated from any monolingual corpus. Specifically, when choosing the sentences A and B for each pretraining example, 50% of the time B is the actual next sentence that follows A, and 50% of the time it is a random sentence from the corpus.

### Pre-training Procedure

#### Training Data

> For the pre-training corpus we use the concatenation of BooksCorpus (800M words) (Zhu et al., 2015) and English Wikipedia (2,500M words). For Wikipedia we extract only the text passages and ignore lists, tables, and headers. It is critical to use a document-level corpus rather than a shuffled sentence-level corpus such as the Billion Word Benchmark (Chelba et al., 2013) in order to extract long contiguous sequences.

#### Training Procedure

> To generate each training input sequence, we sample two spans of text from the corpus, which we refer to as “sentences” even though they are typically much longer than single sentences (but can be shorter also). The first sentence receives the A embedding and the second receives the B embedding. 50% of the time B is the actual next sentence that follows A and 50% of the time it is a random sentence, which is done for the “next sentence prediction” task. They are sampled such that the combined length is ≤ 512 tokens.

> The LM masking is applied after WordPiece tokenization with a uniform masking rate of 15%, and no special consideration given to partial word pieces.

> We train with batch size of 256 sequences (256 sequences * 512 tokens = 128,000 tokens/batch) for 1,000,000 steps, which is approximately 40 epochs over the 3.3 billion word corpus. We use Adam with learning rate of 1e-4, β1 = 0.9, β2 = 0.999, L2 weight decay of 0.01, learning rate warmup over the first 10,000 steps, and linear decay of the learning rate. We use a dropout probability of 0.1 on all layers. We use a **gelu** activation (Hendrycks and Gimpel, 2016) rather than the standard relu, following OpenAI GPT. The training loss is the sum of the mean masked LM likelihood and mean next sentence prediction likelihood

#### Training Resources

>Training of BERTBASE was performed on 4 Cloud TPUs in Pod configuration (16 TPU chips total).5 Training of BERTLARGE was performed on 16 Cloud TPUs (64 TPU chips total). Each pretraining took 4 days to complete.

### Fine tuning procedures.

> For sequence-level classification tasks, BERT fine-tuning is straightforward. In order to obtain a fixed-dimensional pooled representation of the input sequence, we take the final hidden state (i.e., the output of the Transformer) for the first token in the input, which by construction corresponds to the the special [CLS] word embedding. We denote this vector as C ∈ R H. The only new parameters added during fine-tuning are for a classification layer W ∈ R K×H, where K is the number of classifier labels. The label probabilities P ∈ R K are computed with a standard softmax, P = softmax(CWT ). All of the parameters of BERT and W are fine-tuned jointly to maximize the log-probability of the correct label. For spanlevel and token-level prediction tasks, the above procedure must be modified slightly in a taskspecific manner.

> For fine-tuning, most model hyperparameters are the same as in pre-training, with the exception of the batch size, learning rate, and number of training epochs. The dropout probability was always kept at 0.1. The optimal hyperparameter values are task-specific, but we found the following range of possible values to work well across all tasks: 

	 Batch size: 16, 32 
	 Learning rate (Adam): 5e-5, 3e-5, 2e-5 
	 Number of epochs: 3, 4

> We also observed that large data sets (e.g., 100k+ labeled training examples) were far less sensitive to hyperparameter choice than small data sets.

---
---
Resources:

1. [Google BERT blog](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html)

2. [Open Source Code](https://github.com/google-research/bert)

