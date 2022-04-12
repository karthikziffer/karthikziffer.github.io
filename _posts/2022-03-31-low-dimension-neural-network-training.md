---
layout: post
title: "Low dimension neural network training"
author: "Karthik"
categories: journal
tags: [documentation,sample]





---



Deep neural networks can be optimized in randomly-projected subspaces of much smaller dimensionality than their native parameter space. While such training is promising for more efficient and scalable optimization schemes, its practical application is limited by inferior optimization performance.

Here, we improve on recent random subspace approaches as follows: 

- Firstly, we show that keeping the random projection fixed throughout training is detrimental to optimization.

- We propose re-drawing the random subspace at each step, which yields significantly better performance.

<br>

We realize further improvements by applying independent projections to different parts of the network, making the approximation more efficient as network dimensionality grows.

<br>

To implement these experiments, we leverage <mark>hardware-accelerated pseudo-random number generation</mark> to construct the random projections on-demand at every optimization step, allowing us to distribute the computation of independent random directions across multiple workers with shared random seeds.

<br>

This yields significant reductions in memory and is up to 10x faster for the workloads in question.

<br>

<mark>Empirical evidence suggests that not all of the gradient directions are required to sustain effective optimization and that the descent may happen in much smaller subspaces</mark>

<br>

Many methods are able to greatly reduce model redundancy while achieving high task performance at a lower computational cost

<br>

The paper observes that applying smaller independent random projections to different parts of the network and re-drawing them at every step significantly improves the obtained accuracy on fully-connected and several convolutional architectures

<br>

![Capture11.png](https://i.postimg.cc/VL4KxNhf/Capture11.png)

<br>

At the point θ, the black arrow represents the direction of steepest descent computed by conventional SGD. The colored arrow represents the direction of steepest descent under the constraint of being in the chosen lower dimensional random subspace (the green plane)

<br>

<iframe width="560" height="315" src="https://www.youtube.com/embed/eeMJg4uI7o0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>



<br>

To reduce the network training dimensionality, we seek to project into a lower dimensional random subspace by applying a <mark>D*d</mark> random projection matrix <mark>P</mark> to the parameters  

<br>


$$

    \theta_t := \theta_0 + P.c_t
    \\
    \theta_o : Network's \ initialization
    \\
    c_t : Low \ dimensional \ trainable \ parameter \ vector \ of \ size \ d
    \\ 
    d < D
$$


<br>

If <mark>P’s</mark> column vectors are orthogonal and normalized, they form a randomly oriented base and 
$$
c_t - \ can \ be \ interpreted \ as \ coordinates \ in \ the \ subspace.
$$
<br>

As such, the construction can be used to train in a d-dimensional subspace of the network’s original D-dimensional parameter space.

<br>

![Capture11.png](https://i.postimg.cc/Px9jz4m7/Capture11.png)



<br>

In this formulation, however, <mark>any optimization progress is constrained to the particular subspace that is determined by the network initialization and the projection matrix</mark>.

<br>

To obtain a more general expression of subspace optimization, the random projection can instead be formulated as a <mark>constraint of the gradient descent in the original weight space</mark>.

<br>

 The constraint requires the gradient to be expressed in the random base 
$$
g^{(RB)}_t := \sum_{i=1}^{d} c_{i,t} * \varphi_{i,t}
$$
with random basis vectors <br>
$$
\{ \ \varphi_{i,t} \in \mathbb{R}^D \ \}_{i = 1}^{d}
$$


and co-ordinates  <br>
$$
c_{i,t} \in \mathbb{R}
$$


<br>

The gradient step 
$$
g_t^{RB} \in \mathbb{R}^D
$$
 <br>

can be directly used for descent in the native weight space following the standard update equation  
$$
\theta_{t+1} := \theta_t - \eta_{RB} * g_t^{RB}
$$
<br>

To obtain the <mark> d dimensional coordinate vector</mark>, we redefine the training objective itself to implement the random bases constraint
$$
L^{RBD}(c_1,.....,c_d) := L(\theta_t + \sum_{i=1}^{d} c_i * \varphi_{i,t})
$$
<br>

Computing the gradient of this modified objective with respect to 
$$
c = [c_1, ..., c_d]^T \ at \ c = \vec{0}
$$
<br>

and substituting it back into the basis yields a descent gradient that is restricted to the specified set of basis vectors
$$
g_t^{RBD} := \sum_{i=1}^{d} \frac{\partial{L^{RBD}}}{\partial{c_i}}
$$
<br>

This scheme never explicitly calculates a gradient with respect to <mark>θ</mark>, but performs the weight update using only the <mark>d</mark>> coordinate gradients in the respective base.

<br>



---

#### Conclusion:

<br>

- The paper introduced an optimization scheme that restricts gradient descent to a few random directions, re-drawn at every step. 
- This provides further evidence that viable solutions of neural network loss landscape can be found, even if only a small fraction of directions in the weight space are explored. 
- In addition, the paper shows that using compartmentalization to limit the dimensionality of the approximation can further improve task performance.

<br>



