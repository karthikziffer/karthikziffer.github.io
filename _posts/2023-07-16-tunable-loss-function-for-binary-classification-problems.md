---
layout: post
title: "Tunable loss functions for binary classification problems"
author: "Karthik"
categories: journal
tags: [documentation,sample]


---



Paper: [Xtreme Margin: A Tunable Loss Function for Binary Classification Problems](https://arxiv.org/ftp/arxiv/papers/2211/2211.00176.pdf)

<br>

This is a paper summary generated from summarizepaper.com. I edited for better understanding. 

<br>

## Introduction

- Loss functions are crucial in optimizing machine learning algorithms. The choice of loss function impacts the training process and model learning. Binary classification is widely used in various applications. Traditional loss functions for binary classification include binary cross-entropy and hinge loss

<br>

How Xtreme Margin is different

- <mark>Xtreme Margin offers greater flexibility with tunable hyperparameters λ1 and λ2 </mark>

- <mark>Hyperparameters allow users to adjust training based on desired outcomes (precision, AUC score, conditional accuracy)</mark>

- Xtreme Margin is also non-convex and non-differentiable in certain cases where it does not predict correctly. 

- Gradient-based optimization methods may not be directly applicable, alternative techniques like [subgradient optimization](https://web.stanford.edu/class/ee392o/subgrad_method.pdf) can be used

  <br>

  

  ```
  The subgradient method is a simple algorithm for minimizing a nondifferentiable convex function. The method looks very much like the ordinary gradient method for differentiable functions, but with several notable exceptions. For example, the subgradient method uses step lengths that are fixed ahead of time, instead of an exact or approximate line search as in the gradient method. Unlike the ordinary gradient method, the subgradient method is not a descent method; the function value can (and often does) increase.
  
  The subgradient method is far slower than Newton’s method, but is much simpler and can be applied to a far wider variety of problems. By combining the subgradient method with primal or dual decomposition techniques, it is sometimes possible to develop a simple distributed algorithm for a problem. 
  
  
  ```

  

- Xtreme Margin is a promising alternative for binary classification problems. 

<br>

## Formula

Xtreme Margin loss function

<br>
$$
L(y, t_true; \lambda_1, \lambda_2) = \frac{1}{1+ (\sigma(y, y_{true}) + \gamma)}
$$
<br>



<br>
$$
\gamma = \ \  1_[ytrue = ypred \ \  \& \ \ ytrue = 0] \ * \ \lambda_1 (2y - 1)^2 + \ \  1_[ytrue = ypred \ \  \& \ \ ytrue = 1] * \ \lambda_2 (2y - 1)^2
$$


<br>

\lambda_1 (2y - 1)^2  term of the expression below is the <mark>extreme margin term</mark>, and is derived from the squared difference between the true conditional probability prediction score of belonging to the default class and the true conditional probability prediction score of belonging to the non-default class. 

<br>
$$
1_A (x) := \begin{cases} 1 \ if \ x \in A  \\ 0 \ if \ x \notin A \end{cases}
$$
<br>

<br>
$$
\sigma(y, y_{true}) := \begin{cases} 0 \ \ if \ \ |y - y_{true}| \ \ < \ \ 0.5 \\  \frac{1}{e|y_{true} - y|} - 1 \end{cases}
$$
<br>



<br>
$$
y_{pred} := \begin{cases} 1 \ if y \ge 0.50 \\ 0 \ if \ y \le 0.50   \end{cases}
$$


<br>



## Tensorflow implementation

<iframe   src="https://carbon.now.sh/embed?bg=rgba%28171%2C+184%2C+195%2C+1%29&t=monokai&wt=none&l=auto&width=808&ds=true&dsyoff=20px&dsblur=68px&wc=true&wa=false&pv=56px&ph=56px&ln=false&fl=1&fm=Hack&fs=14px&lh=133%25&si=false&es=2x&wm=false&code=import%2520tensorflow%2520as%2520tf%250Afrom%2520keras%2520import%2520backend%2520as%2520K%250A%250Almbda1%2520%253D%2520%2523%2523%2523%2520User-defined%2520hyperparameter%250Almbda2%2520%253D%2520%2523%2523%2523%2520User-defined%2520hyperparameter%250A%250Adef%2520indicator1%28y_true%252C%2520y_pred%29%253A%250A%2520%2520%2520%2520if%2520tf.equal%28tf.dtypes.cast%28y_true%252C%2520tf.float32%29%252C%2520y_pred%29%2520and%2520tf.equal%28tf.dtypes.cast%28y_true%252C%2520tf.float32%29%252C%2520tf.constant%280.%29%29%253A%250A%2520%2520%2520%2520%2520%2520%2509return%2520tf.constant%281.%29%250A%2520%2520%2520%2520else%253A%2520return%2520tf.constant%280.%29%250A%250Adef%2520indicator2%28y_true%252C%2520y_pred%29%253A%250A%2520%2520%2520%2520if%2520tf.equal%28tf.dtypes.cast%28y_true%252C%2520tf.float32%29%252C%2520y_pred%29%2520and%2520tf.equal%28tf.dtypes.cast%28y_true%252C%2520tf.float32%29%252C%2520tf.constant%280.%29%29%253A%250A%2520%2520%2520%2520%2520%2520%2509return%2520tf.constant%281.%29%250A%2520%2520%2520%2520else%253A%2520return%2520tf.constant%280.%29%250A%2520%2520%2520%2520%250Adef%2520sigma%28y%252C%2520y_true%29%253A%250A%2520%2520%2520%2520if%2520tf.less%28tf.abs%28tf.subtract%28y%252C%2520tf.dtypes.cast%28%250A%2520%2520%2520%2520%2520%2520%2520%2520y_true%252C%2520tf.float32%29%29%29%252C%25200.5%29%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520out%2520%253D%25200.%250A%2520%2520%2520%2520else%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520out%2520%253D%2520tf.subtract%28tf.divide%281%252C%2520tf.multiply%28%250A%2520%2520%2520%2520%2520%2520%2520%25202.718%252C%2520tf.abs%28tf.subtract%28tf.dtypes.%250A%2520%2520%2520%2520%2520%2520%2520%2520cast%28y_true%252C%2520tf.float32%29%252C%2520y%29%29%29%29%252C%25201.%29%250Areturn%2520out%250A%250A%250Adef%2520xtreme_margin_loss%28y_true%252C%2520y%29%253A%250A%2520%2520%2520%2520y_pred%2520%253D%2520tf.reshape%28tf.constant%281.%29%252C%2520%255B1%252C1%255D%29%250A%2520%2520%2520%2520%2520%2520%2520%2520if%2520tf.equal%28tf.greater%28y%252C%2520tf.constant%280.5%29%29%252C%2520True%29%2520else%2520tf.reshape%28tf.constant%280.%29%252C%2520%255B1%252C1%255D%29%250A%2520%2520%2520%2520%2509loss%2520%253D%2520tf.divide%281.%252C%2520tf.add%281.%252C%2520tf.add%28sigma%28y%252C%2520y_true%29%252C%2520tf.add%28tf.multiply%28indicator1%28y_true%252C%2520y_pred%29%252C%2520tf.multiply%28lmbda1%252C%250A%2520%2520%2520%2520%2520%2520%2520%2520tf.square%28tf.subtract%28tf.multiply%282.%252C%2520y%29%252C%25201.%29%29%29%29%252C%2520tf.multiply%28indicator2%28y_true%252C%2520y_pred%29%252C%2520tf.multiply%28lmbda2%252C%2520tf.square%28%2520tf.subtract%28tf.multiply%282.%252C%2520y%29%252C%25201.%29%29%29%29%29%29%29%29%250A%2520%2520%2520%2520return%2520K.mean%28loss%252C%2520axis%253D-1%29"   style="width: 808px; height: 912px; border:0; transform: scale(1); overflow:hidden;"   sandbox="allow-scripts allow-same-origin"> </iframe>



<br>

## Conclusion

On the Ionosphere dataset used for our experiment, even though the binary cross-entropy loss function achieved a higher mean cross-validation accuracy compared to the Xtreme Margin loss function, its conditional accuracy cannot be manually controlled, as it is internally chosen during the training process on the loss function. In some situations, it suffers from a low conditional accuracy for one or both classes.

The tunable component of Xtreme Margin enables practitioners to choose whether they want to maximize precision or recall. Since there is a tradeoff between precision and recall (as the precision increases, the recall decreases and vice versa), one has to place more emphasis on a particular metric depending on the use case.







