---
layout: post
title: "Feed Forward Propagation"
author: "Karthik"
categories: journal
tags: [documentation,sample]
<!-- image: cards.jpg -->
---






$$
a^{(2)}_1 = g( \theta^{(1)}_{10} x_0  + \theta^{(1)}_{11} x_1 + \theta^{(1)}_{12} x_2 + \theta^{(1)}_{13} x_3)
$$

$$
a^{(2)}_2 = g( \theta^{(1)}_{20} x_0  + \theta^{(1)}_{21} x_1 + \theta^{(1)}_{22} x_2 + \theta^{(1)}_{23} x_3)
$$

$$
a^{(2)}_3 = g( \theta^{(1)}_{30} x_0  + \theta^{(1)}_{31} x_1 + \theta^{(1)}_{32} x_2 + \theta^{(1)}_{33} x_3)
$$

$$
h_{\theta}(x) = g( \theta_{10}^{(2)} a_{(0)}^{(2)} + \theta_{11}^{(2)} a_{(1)}^{(2)} + \theta_{12}^{(2)} a_{(2)}^{(2)} + \theta_{13}^{(2)} a_{(3)}^{(2)} )
$$



### We replace  

$$
z^{(2)}_1 = \theta^{(1)}_{10} x_0  + \theta^{(1)}_{11} x_1 + \theta^{(1)}_{12} x_2 + \theta^{(1)}_{13} x_3
$$



### Then ,

$$
a_1^{(2)} = g(z_1^{(2)})
$$

$$
a_2^{(2)} = g(z_2^{(2)})
$$

$$
a_3^{(2)} = g(z_3^{(2)})
$$



