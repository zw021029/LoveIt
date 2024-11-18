---
title: "Machine Learning by Prof. Zhou"
subtitle: ""
date: 2024-11-12T19:56:49+08:00
lastmod: 2024-11-12T19:56:49+08:00
draft: false
author: "Wei"
authorLink: ""
description: ""
license: ""
images: []

tags: []
categories: ["Original of machine learning"]

featuredImage: ""
featuredImagePreview: ""

hiddenFromHomePage: false
hiddenFromSearch: false
twemoji: false
lightgallery: true
ruby: true
fraction: true
fontawesome: true
linkToMarkdown: true
rssFullText: false

toc:
  enable: true
  auto: true
code:
  copy: true
  maxShownLines: 50
math:
  enable: true
  # ...
mapbox:
  # ...
share:
  enable: true
  # ...
comment:
  enable: true
  # ...
library:
  css:
    # someCSS = "some.css"
    # located in "assets/"
    # Or
    # someCSS = "https://cdn.example.com/some.css"
  js:
    # someJS = "some.js"
    # located in "assets/"
    # Or
    # someJS = "https://cdn.example.com/some.js"
seo:
  images: []
  # ...
---

<!--more-->

## 0. Lead-in
- These are my notes, reflections and thoughts while reading Professor Zhou Zhihua's book Machine Learning (Tsinghua University Press).
- Thanks to this book, which inspired my doctoral work and career development.

&nbsp; &nbsp; &nbsp; &nbsp; Regarding this topic, I have several follow-up questions. Three years ago, during my first year in college, I started learning from the basics like decision trees, support vector machines, and ensemble learning; later, I progressed to deep learning, including DNN, BNN, CNN, LSTM, GRU, GNN, as well as transfer learning, active learning, reinforcement learning, and domain adaptation. Over the years, models like KAN, Transformer, BERT, and ChatGPT have also gained popularity. Nowadays, any discussion on AI almost inevitably involves large models, big data, and massive computing power, flourishing across nearly every field, making it seem like a "cure-all." \
&nbsp; &nbsp; &nbsp; &nbsp; But is this truly the trend for the future? As the "No Free Lunch Theorem" suggests, it is meaningless to discuss a model’s universal efficacy without considering the “hypothesis space” of the problem. I believe that large models are no exception to this rule: the more we emphasize a tool's universal applicability, the more caution we should exercise. \
&nbsp; &nbsp; &nbsp; &nbsp; From the trajectory of AI development, it seems that the current trend of "large models" has a certain allure that may be distracting. Moreover, there are several critical issues that remain unresolved:

1. For large models themselves, what exactly is their hypothesis space? Is the special phenomenon of “illusion” truly a result of reaching certain thresholds in data volume and parameter count? Is there a causal relationship between these two factors, or merely a phenomenal correlation? If it is indeed causative, then, philosophically speaking, what are the conditions for this shift from quantitative to qualitative change? If we use an infinite amount of data with enough neurons and parameters, would our world be merely one dimension within countless hallucinated parallel worlds?
2. Regarding the future trends of large models, is it truly the case that more data and more parameters yield better outcomes? Can this outcome genuinely be called “intelligence”? What is our definition of “intelligence” in the context of “artificial intelligence”? Are the neural networks of the brain and those of machines merely structurally similar, or do they share something deeper?
3. If a simple model or a single algorithm can solve a problem, is there still a need for large models?

## 1. Introduction
### Topic One: What is the best algorithm of machine learning? 
- **Inductive preference**
  - Any effective machine learning algorithm has an *inductive preference*.
  - every machine learning should be able to make heuristic selection and induction of hypotheses from a huge hypothesis space, and then form so-called **"knowledge"**.
  - Otherwise, machine learning algorithms are meaningless.
- **No Free Lunch Theorem**
  -



## 2. Neural Networks 
### Topic One: Neuron, Perception and Multi-layer perceptron


### Topic Two: Error back propagation algorithm
### **Topic Two: Error back propagation algorithm**

#### **Mathematical derivation**

Assume there is a simple neural network with **one input layer**, **one hidden layer**, and **one output layer**. The input layer has $d$ neurons, the hidden layer has $q$ neurons, and the output layer has $l$ neurons. The activation functions of the hidden layer and output layer are Sigmoid functions. The loss function is $L(y, \hat{y})$.

---

##### **1. Forward Propagation Formula**

1. **Hidden Layer**:
   $$
   z_j^1 = \sum_{i=1}^{d} w_{ij}^1 x_i + b_j^1, \quad j = 1, 2, \cdots, q
   $$
   $$
   a_j^1 = \sigma(z_j^1), \quad j = 1, 2, \cdots, q
   $$

2. **Output Layer**:
   $$
   z_j^2 = \sum_{i=1}^{q} w_{ij}^2 a_i^1 + b_j^2, \quad j = 1, 2, \cdots, l
   $$
   $$
   a_j^2 = \sigma(z_j^2), \quad j = 1, 2, \cdots, l
   $$

---

##### **2. Loss Function**

The loss function is the binary cross-entropy loss:
$$
L(y, \hat{y}) = - \sum_{j=1}^{l} y_j \log \hat{y}_j + (1 - y_j) \log (1 - \hat{y}_j)
$$
where:
- $y_j$ is the ground truth label,
- $\hat{y}_j = a_j^2$ is the predicted value (activation of the output layer).

---

##### **3. Back Propagation Formula**

###### **3.1. Output Layer Error Term**
The error term for the output layer is:
$$
\delta_j^2 = \frac{\partial L}{\partial z_j^2} = \frac{\partial L}{\partial a_j^2} \cdot \frac{\partial a_j^2}{\partial z_j^2}
$$
1. **Loss Function Derivative**:
   $$
   \frac{\partial L}{\partial a_j^2} = -\frac{y_j}{a_j^2} + \frac{1 - y_j}{1 - a_j^2}
   $$
2. **Sigmoid Derivative**:
   $$
   \sigma'(z_j^2) = \sigma(z_j^2) (1 - \sigma(z_j^2))
   $$
3. **Combine**:
   $$
   \delta_j^2 = \left( -\frac{y_j}{a_j^2} + \frac{1 - y_j}{1 - a_j^2} \right) \cdot a_j^2 (1 - a_j^2)
   $$

###### **3.2. Hidden Layer Error Term**
The error term for the hidden layer is:
$$
\delta_j^1 = \frac{\partial L}{\partial z_j^1} = \frac{\partial L}{\partial a_j^1} \cdot \frac{\partial a_j^1}{\partial z_j^1}
$$
1. **Gradient of $a_j^1$**:
   $$
   \frac{\partial L}{\partial a_j^1} = \sum_{k=1}^{l} \frac{\partial L}{\partial z_k^2} \cdot \frac{\partial z_k^2}{\partial a_j^1}
   $$
   Since:
   $$
   \frac{\partial z_k^2}{\partial a_j^1} = w_{kj}^2
   $$
   We get:
   $$
   \frac{\partial L}{\partial a_j^1} = \sum_{k=1}^{l} \delta_k^2 w_{kj}^2
   $$
2. **Sigmoid Derivative**:
   $$
   \sigma'(z_j^1) = \sigma(z_j^1) (1 - \sigma(z_j^1))
   $$
3. **Combine**:
   $$
   \delta_j^1 = \left( \sum_{k=1}^{l} \delta_k^2 w_{kj}^2 \right) \cdot \sigma'(z_j^1)
   $$

---

##### **4. Gradient Calculation**

1. **Output Layer Gradients**:
   - For weights:
     $$
     \frac{\partial L}{\partial w_{ij}^2} = \delta_j^2 \cdot a_i^1
     $$
   - For biases:
     $$
     \frac{\partial L}{\partial b_j^2} = \delta_j^2
     $$

2. **Hidden Layer Gradients**:
   - For weights:
     $$
     \frac{\partial L}{\partial w_{ij}^1} = \delta_j^1 \cdot x_i
     $$
   - For biases:
     $$
     \frac{\partial L}{\partial b_j^1} = \delta_j^1
     $$

---

##### **5. Parameter Update Formula**

Using gradient descent with learning rate $\eta$, the parameters are updated as:
$$
\begin{aligned}
& w_{ij}^2 \leftarrow w_{ij}^2 - \eta \frac{\partial L}{\partial w_{ij}^2} \\
& b_j^2 \leftarrow b_j^2 - \eta \frac{\partial L}{\partial b_j^2} \\
& w_{ij}^1 \leftarrow w_{ij}^1 - \eta \frac{\partial L}{\partial w_{ij}^1} \\
& b_j^1 \leftarrow b_j^1 - \eta \frac{\partial L}{\partial b_j^1}
\end{aligned}
$$

---

##### **6. Conclusion**

The back propagation algorithm is a gradient descent-based approach for minimizing the neural network's loss function. It includes:
1. **Forward propagation**: Calculate activations for each layer.
2. **Backward propagation**: Compute the error terms and gradients for each layer.
3. **Parameter updates**: Adjust weights and biases using the computed gradients.

