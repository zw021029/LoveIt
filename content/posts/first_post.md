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

# 0. Lead-in
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


## 2. Neural Networks 
### Topic One: Neuron, Perception and Multi-layer perceptron

### Topic Two: Error back propagation algorithm

Assume there is a simple neural network with **one input layer**, **one hidden layer**, and **one output layer**. The input layer has $d$ neurons, the hidden layer has $q$ neurons, and the output layer has $l$ neurons. The activation functions of the hidden layer and output layer are Sigmoid functions. The loss function is $L(y, \hat{y})$.

---

#### **1. Forward Propagation Formula**

Forward propagation calculates the activations of each layer step by step, using the weights, biases, and the activation function.

1. **Hidden Layer**:
   - Compute the weighted sum of inputs and biases:
     $$
     z_j^1 = \sum_{i=1}^{d} w_{ij}^1 x_i + b_j^1, \quad j = 1, 2, \cdots, q
     $$
   - Apply the Sigmoid activation function:
     $$
     a_j^1 = \sigma(z_j^1), \quad j = 1, 2, \cdots, q
     $$

2. **Output Layer**:
   - Compute the weighted sum of hidden layer outputs and biases:
     $$
     z_j^2 = \sum_{i=1}^{q} w_{ij}^2 a_i^1 + b_j^2, \quad j = 1, 2, \cdots, l
     $$
   - Apply the Sigmoid activation function:
     $$
     a_j^2 = \sigma(z_j^2), \quad j = 1, 2, \cdots, l
     $$
   - **Note**: The activation values $a_j^2$ at the output layer are the predicted probabilities $\hat{y}_j$.

---

#### **2. Loss Function**

The binary cross-entropy loss function quantifies the difference between predicted values $\hat{y}$ and true labels $y$. It is defined as:
$$
L(y, \hat{y}) = - \sum_{j=1}^{l} y_j \log \hat{y}_j + (1 - y_j) \log (1 - \hat{y}_j)
$$

**Thinking**:
- This loss function penalizes incorrect predictions by comparing the predicted probabilities $\hat{y}_j$ with the true labels $y_j$.
- The logarithmic form ensures that the model is more heavily penalized for confident but incorrect predictions.

---

#### **3. Back Propagation Formula**

Back propagation computes the gradients of the loss function with respect to the weights and biases of each layer. It involves:
1. Calculating the **error terms** for each layer (how much each layer contributes to the loss).
2. Using the error terms to calculate the gradients of the weights and biases.

##### **3.1. Output Layer Error Term**

The error term $\delta_j^2$ for the output layer is:
$$
\delta_j^2 = \frac{\partial L}{\partial z_j^2} = \frac{\partial L}{\partial a_j^2} \cdot \frac{\partial a_j^2}{\partial z_j^2}
$$

1. **Loss Function Derivative**:
   $$
   \frac{\partial L}{\partial a_j^2} = -\frac{y_j}{a_j^2} + \frac{1 - y_j}{1 - a_j^2}
   $$
   - This represents how the output probabilities $\hat{y}_j$ affect the loss.

2. **Sigmoid Derivative**:
   $$
   \sigma'(z_j^2) = \sigma(z_j^2) (1 - \sigma(z_j^2))
   $$
   - The derivative of the Sigmoid function expresses how the weighted input $z_j^2$ affects the output.

3. **Combine**:
   $$
   \delta_j^2 = \left( -\frac{y_j}{a_j^2} + \frac{1 - y_j}{1 - a_j^2} \right) \cdot a_j^2 (1 - a_j^2)
   $$

##### **3.2. Hidden Layer Error Term**

The error term $\delta_j^1$ for the hidden layer is:
$$
\delta_j^1 = \frac{\partial L}{\partial z_j^1} = \frac{\partial L}{\partial a_j^1} \cdot \frac{\partial a_j^1}{\partial z_j^1}
$$

1. **Gradient of $a_j^1$**:
   $$
   \frac{\partial L}{\partial a_j^1} = \sum_{k=1}^{l} \frac{\partial L}{\partial z_k^2} \cdot \frac{\partial z_k^2}{\partial a_j^1}
   $$
   - The gradient $\frac{\partial L}{\partial z_k^2}$ comes from the output layer error term $\delta_k^2$.
   - $\frac{\partial z_k^2}{\partial a_j^1} = w_{kj}^2$ indicates how weights in the output layer affect the loss.

2. **Sigmoid Derivative**:
   $$
   \sigma'(z_j^1) = \sigma(z_j^1) (1 - \sigma(z_j^1))
   $$

3. **Combine**:
   $$
   \delta_j^1 = \left( \sum_{k=1}^{l} \delta_k^2 w_{kj}^2 \right) \cdot \sigma'(z_j^1)
   $$

---

#### **4. Gradient Calculation**

Using the error terms, calculate the gradients of weights and biases.

1. **Output Layer Gradients**:
   - For weights:
     $$
     \frac{\partial L}{\partial w_{ij}^2} = \delta_j^2 \cdot a_i^1
     $$
     - This represents how changes in weights affect the loss through the output layer.
   - For biases:
     $$
     \frac{\partial L}{\partial b_j^2} = \delta_j^2
     $$
     - This represents how changes in biases affect the loss.

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

#### **5. Parameter Update Formula**

Using gradient descent with learning rate $\eta$, update weights and biases:
$$
\begin{aligned}
& w_{ij}^2 \leftarrow w_{ij}^2 - \eta \frac{\partial L}{\partial w_{ij}^2} \\
& b_j^2 \leftarrow b_j^2 - \eta \frac{\partial L}{\partial b_j^2} \\
& w_{ij}^1 \leftarrow w_{ij}^1 - \eta \frac{\partial L}{\partial w_{ij}^1} \\
& b_j^1 \leftarrow b_j^1 - \eta \frac{\partial L}{\partial b_j^1}
\end{aligned}
$$

---

#### **6. Key Insights**

1. **Sigmoid function and gradients**:
   - Sigmoid ensures outputs are in the range $(0, 1)$, making it suitable for binary classification.
   - However, its derivative can lead to **vanishing gradients** when values are close to $0$ or $1$.

2. **Error propagation**:
   - Errors from the output layer propagate backward to update the hidden layer. This ensures all layers contribute to minimizing the loss.

3. **Learning rate ($\eta$)**:
   - A small $\eta$ ensures stable convergence but can slow down training.
   - A large $\eta$ may cause the training to overshoot and fail to converge.

4. **Cross-entropy loss**:
   - It is preferred for classification tasks because it directly measures the difference between predicted probabilities and true labels.

---

#### **7. Conclusion**

The back propagation algorithm is a gradient descent-based approach to minimize the neural network's loss function. Its process involves:
1. **Forward propagation**: Calculate activations for each layer.
2. **Backward propagation**: Compute error terms and gradients for each layer.
3. **Parameter updates**: Adjust weights and biases iteratively to minimize the loss.

Through these steps, the neural network learns to make better predictions over time.
