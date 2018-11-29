---
layout: post
title: "Summary - SGDR"
author: "Karthik"
categories: journal
tags: [documentation,sample]
---




[SGDR: STOCHASTIC GRADIENT DESCENT WITH WARM RESTARTS](https://arxiv.org/pdf/1608.03983.pdf)


>Restart techniques are common in gradient-free optimization to deal with multimodal functions. Partial warm restarts are also gaining popularity in gradientbased optimization to improve the rate of convergence in accelerated gradient schemes to deal with ill-conditioned functions.

> In this paper, we propose a simple warm restart technique for stochastic gradient descent to improve its anytime performance when training deep neural networks.

>The training of a DNN with **n** free parameters can be formulated as the problem of minimizing a function 
f : $$R^n$$ → $$R$$. 

> The commonly used procedure to optimize f is to iteratively adjust
>  $$x_t$$ ∈ $$R^n$$ (the parameter vector at time step t) using gradient information $$∇f_t(x_t)$$ obtained on a relatively small t-th batch of b datapoints. The Stochastic Gradient Descent (SGD) procedure then becomes an extension of the Gradient Descent (GD) to stochastic optimization of f as follows:

$$x_{t+1} = x_t − η_t∇f_t(x_t)$$

GSD with momentum is most widely used.


$$v_{i+1} = \mu_t v_t - \nabla f_t(x_t)$$
<br>
$$x_{t+1} = x_t + v_{t+1}$$


> where $$v_t$$ is a velocity vector initially set to 0, $$η_t$$ is a decreasing learning rate and $$µ_t$$ is a momentum rate which defines the trade-off between the current and past observations of $$∇f_t(x_t)$$.

> The main difficulty in training a DNN is then associated with the scheduling of the learning rate and the amount of L2 weight decay regularization employed.

>A common learning rate schedule is to use a constant learning rate and divide it by a fixed constant in (approximately) regular intervals.


What method is followed in the Paper
>In this paper, we propose to periodically simulate warm restarts of SGD, where in each restart the learning rate is initialized to some value and is scheduled to decrease.


Important take away
> Our empirical results suggest that SGD with warm restarts requires 2× to 4× fewer epochs than the currently-used learning rate schedule schemes to achieve comparable or even better results.


Procedure caried out
> In this work, we consider one of the simplest warm restart approaches. We simulate a new warmstarted run / restart of SGD once $$T_i$$ epochs are performed, where i is the index of the run. 

>Importantly, the restarts are not performed from scratch but emulated by increasing the learning rate $$η_t$$ while the old value of $$x_t$$ is used as an initial solution.

> The amount of this increase controls to which extent the previously acquired information (e.g., momentum) is used.

---
Let's see some math

Within the i-th run, we decay the learning rate with a cosine annealing for each batch as follows:

$$η_t = η ^i_ {min} + \frac{1}{2} (η ^i_{ max} − η ^i_{ min})(1 + cos(\frac{T_{cur}} {T_i} π))$$

- $$η^i_{min}$$ and $$η^i_{max}$$ are ranges for the learning rate.
- $$T_{cur}$$ accounts for how many epochs have been performed since the last restart. Since $$T_{cur}$$ is updated at each batch iteration **t**.

---
---
Understanding $$η_t$$:

   $$η_t = η^i_{max}   \ \ when   \ \  t = 0 \  \ and  \   \ T_{cur} = 0$$
   $$η_t = η^i_{min}  \ \  when  \ \  T_{cur} =  T_i , \ \ cos{\pi} = (-1)$$

---
---
> In order to improve anytime performance, we suggest an option to start with an initially small Ti and increase it by a factor of Tmult at every restart.

> . It might be of great interest to decrease $$η^i_{max}$$ and $$η^i_{min}$$ at every new restart. However, for the sake of simplicity, here, we keep $$η^i_{max}$$ and $$η^i_{min}$$ the same for every **i** to reduce the number of hyperparameters involved.

>Since our simulated warm restarts (the increase of the learning rate) often temporarily worsen performance, we do not always use the last $x_t$ as our recommendation for the best solution (also called the **incumbent solution**).


Conclusion 

During initial phase of network training the learning rate should be high , As the training continues, the learning rate should be reduced to avoid over shooting away from global minima. SGDR reduces the learning rate as the training persists. Restarting feature in SGDR saves the network from getting struck at local minima. 