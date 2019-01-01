---
layout: post
title: "Gated Graph Sequence Neural Networks"
author: "Karthik"
categories: journal
tags: [documentation,sample]
---






<br>

[Gated Graph Sequence Neural Networks](https://arxiv.org/pdf/1511.05493.pdf)

---

<br>

Authors: 

- Yujia Li & Richard Zemel 
  * Department of Computer Science, University of Toronto 
  * Toronto, Canada 
  * {yujiali,zemel}@cs.toronto.edu 
- Marc Brockschmidt & Daniel Tarlow 
  * Microsoft Research Cambridge, 
  * UK 
  * {mabrocks,dtarlow}@microsoft.com

---



<br>

#### Abstract

- Graph-structured data appears frequently in domains including chemistry, natural language semantics, social networks, and knowledge bases. In this work, we study feature learning techniques for graph-structured inputs.

- Our starting point is previous work on *Graph Neural Networks (Scarselli et al., 2009)*, which we modify to use gated recurrent units and modern optimization techniques and then extend to output sequences.

- The result is a flexible and broadly useful class of neural network models that has favorable inductive biases relative to purely sequence-based models (e.g., LSTMs) when the problem is graph-structured.

- The paper demonstrate the capabilities on some simple AI (bAbI) and graph algorithm learning tasks. We then show it achieves state-of-the-art performance on a problem from program verification, in which subgraphs need to be described as abstract data structures.

---



<br>



#### Introduction

- Our main contribution is an extension of Graph Neural Networks that outputs sequences. Previous work on feature learning for graph-structured inputs has focused on models that produce single outputs such as graph-level classifications, but many problems with graph inputs require outputting sequences.

- Examples include paths on a graph, enumerations of graph nodes with desirable properties, or sequences of global classifications mixed with, for example, a start and end node. We are not aware of existing graph feature learning work suitable for this problem. 
- Our motivating application comes from program verification and requires outputting logical formulas, which we formulate as a sequential output problem. 
- A secondary contribution is highlighting that Graph Neural Networks (and further extensions we develop here) are a broadly useful class of neural network model that is applicable to many problems currently facing the field.

<br>



**There are two settings for feature learning on graphs:**

- [ ] Learning a representation of the input graph. 
- [ ] Learning representations of the internal state during the process of producing a sequence of outputs.



---

<br>

- **Learning a representation of the input graph** is mostly achieved by previous work on *Graph Neural Networks (Scarselli et al., 2009)*. we make several minor adaptations of this framework, including changing it to use modern practices around Recurrent Neural Networks. 
- **Learning representations of the internal state during the process of producing a sequence of outputs** is important because we desire outputs from graphstructured problems that are not solely individual classifications. 
- In these cases, the challenge is how to learn features on the graph that encode the partial output sequence that has already been produced (e.g., the path so far if outputting a path) and that still needs to be produced (e.g., the remaining path). We will show how the GNN framework can be adapted to these settings, leading to a novel graph-based neural network model that we call Gated Graph Sequence Neural Networks (GGS-NNs).



<br>



We illustrate aspects of this general model in experiments on *bAbI tasks (Weston et al., 2015)* and graph algorithm learning tasks that illustrate the capabilities of the model.

We then present an application to the verification of computer programs. When attempting to prove properties such as memory safety (i.e., that there are no null pointer dereferences in a program), a core problem is to find mathematical descriptions of the data structures used in a program.

Following *Brockschmidt et al. (2015)*, we have phrased this as a machine learning problem where we will learn to map from a set of input graphs, representing the state of memory, to a logical description of the data structures that have been instantiated. Whereas *Brockschmidt et al. (2015)* relied on a large amount of hand-engineering of features, we show that the system can be replaced with a GGS-NN at no cost in accuracy.

<br>

---

<br>

**Let us cover all three kinds of Networks.**

1.  Graph Neural Networks.
2.  Gated Graph Neural Networks.
3.  Gated Graph Sequence Neural Networks.

---



<br>

#### Graph Neural Networks:

- GNNs are a general neural network architecture defined according to a graph structure G = (V, E) .  V - Nodes , E - Edges
- GNNs have nodes , edges. The edges are pairs. This work is focused on directed graphs. But this framework can be adapted to undirected graphs too.
- Each node has a node vector/representation/embedding.
- Graphs contain node labels for each node and edge labels for each edge.
- Set of predecessor nodes and set of successor nodes with edges is obtained from dedicated function.
- Set of all nodes neighboring is obtained from the union predecessor function output and successor function output.
- GNNs map graphs to outputs via two steps. 
  - First, there is a **propagation step** that computes node representations for each node.
  - Second, an **output model** maps from node representations
    and corresponding labels to an output.



##### Propagation Model 

- In this model, An iterative procedure propagates node representations.
- Initial node representations are set to arbitrary values, then each node representation is updated following the recurrence until convergence.

##### Output Model and Learning

- The output model is defined per node and is a differentiable function that maps to an output.
- This is generally a linear or neural network mapping.

- Learning is done via the *Almeida-Pineda algorithm (Almeida, 1990; Pineda, 1987)*, which works by running the propagation to convergence, and then computing gradients based upon the converged solution.
- This has the advantage of not needing to store intermediate states in order to compute gradients.
- The disadvantage is that parameters must be constrained so that the propagation step is a **contraction map**.
- This is needed to ensure convergence, but it may limit the expressivity of the
  model.





---



#### GATED GRAPH NEURAL NETWORKS:



- We describe Gated Graph Neural Networks (GG-NNs), our adaptation of GNNs that is suitable for non-sequential outputs.
- The biggest modification of GNNs is that we use *Gated Recurrent Units (Cho et al., 2014)* and unroll the recurrence for a fixed number of steps **T** and use backpropagation through time in order to compute gradients.
- This requires more memory than the *Almeida-Pineda algorithm*, but it removes the need to constrain parameters to ensure convergence. We also extend the underlying representations and output model.



##### Node Annotations.

- In GNNs, there is no point in initializing node representations because the contraction map constraint ensures that the fixed point is independent of the initializations.
- This is no longer the case with GG-NNs, which lets us incorporate node labels as additional inputs.
- To distinguish these node labels used as inputs from the ones introduced before, we call them **Node Annotations**, and use vector **x** to denote these annotations.



##### Propagation Model



- The Mathematical equation for basic recurrence of the propagation model is given in the [Paper](https://arxiv.org/pdf/1511.05493.pdf)

- The Parameter tying and sparsity is constructed as a recurrent matrix **A**.

- The matrix A determines how nodes in the graph communicate with each other.

- The sparsity structure corresponds to the edges of the graph, and the parameters in each submatrix are determined by the edge type
  and direction.

- **A** contain two columns of blocks i.e Outgoing edges and Incoming edges.

- The initialization step, which copies node annotations into the first components of the hidden state and pads the rest with zeros.

- Then the information is passed between different nodes of the graph via incoming and outgoing edges with parameters dependent on the edge type and direction.

- > The remaining are GRU-like updates that incorporate information from the other nodes and from the previous timestep to update each node’s hidden state. z and r are the update and reset gates, logistic sigmoid function. 
  >
  >
  >
  > We initially experimented with a vanilla recurrent neural network-style update, but in preliminary experiments we found this GRU-like
  > propagation step to be more effective.







##### Output Models

- There are several types of one-step outputs that we would like to produce in different situations. First, GG-NNs support node selection tasks for each node and output node scores and applying a softmax over node scores.
- Second for graph level outputs , we define a graph level representation vector.





---



#### GATED GRAPH SEQUENCE NEURAL NETWORKS:



- Gated Graph Sequence Neural Networks (GGS-NNs) contains several GG-NNs
  operate in sequence to produce an output sequence.

- > **Sequence outputs with observed annotations** Consider the task of making a sequence of predictions for a graph, where each prediction is only about a part of the graph. In order to ensure we predict an output for each part of the graph exactly once, it suffices to have one bit per node, indicating whether the node has been “explained” so far. In some settings, a small number of annotations are sufficient to capture the state of the output procedure. When this is the case, we may want to directly input this information into the model via labels indicating target intermediate annotations. In some cases, these annotations may be sufficient, in that we can define a model where the GG-NNs are rendered conditionally independent given the annotations.



  > In this case, at training time, given the annotations the sequence prediction task decomposes into single step prediction tasks and can be trained as separate GG-NNs. At test time, predicted annotations from one step will be used as input to the next step. This is analogous to training directed graphical models when data is fully observed.

- > **Sequence outputs with latent annotations** More generally, when intermediate node annotations are not available during training, we treat them as hidden units in the network, and train the whole model jointly by backpropagating through the whole sequence.





---



#### Applications:



- >  **BABI Tasks**:
  >
  > >  The bAbI tasks are meant to test reasoning capabilities that AI systems should be capable of. In the
  > > bAbI suite, there are 20 tasks that test basic forms of reasoning like deduction, induction, counting,
  > > and path-finding.

- > **PROGRAM VERIFICATION WITH GGS-NNS:**
  >
  > > GGS-NNs are motivated by a practical application in program verification. A crucial step in automatic program verification is the inference of program invariants, which approximate the set of program states reachable in an execution. Finding invariants about data structures is an open problem.





For more information and Mathematical calculations please refer [GATED GRAPH SEQUENCE NEURAL NETWORKS Paper](https://arxiv.org/pdf/1511.05493.pdf)





---





