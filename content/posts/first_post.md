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
  enable: false
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
- **NFL**

