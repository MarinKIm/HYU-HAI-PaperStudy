# ResNet Paper Study

Study notes and implementation of the paper:

Deep Residual Learning for Image Recognition
CVPR 2016

Paper: https://arxiv.org/abs/1512.03385

---

# Overview

Deep neural networks tend to perform better as depth increases.
However, simply stacking more layers often leads to **degradation**, where deeper networks produce higher training error.
<img width="579" height="205" alt="Image" src="https://github.com/user-attachments/assets/a948a004-c9ac-44c7-ae2e-6f7069e55d12" />

This paper introduces **Residual Learning**, a technique that allows very deep networks to be trained effectively.

The key idea is to learn a residual function instead of the original mapping.

H(x) → F(x) = H(x) − x

Thus the output becomes:

y = F(x) + x

This simple reformulation significantly improves optimization for deep networks.

---

# Degradation Problem

Empirical observations show:

18-layer network → lower training error
34-layer network → higher training error

This problem is **not caused by overfitting**, but by optimization difficulty.

Residual learning addresses this issue by introducing **skip connections**.

---

# Residual Block

Core structure of ResNet:

Input x
↓
Conv
↓
BatchNorm
↓
ReLU
↓
Conv
↓
BatchNorm
↓
Add skip connection
↓
ReLU

The skip connection allows the network to directly propagate information and gradients.

---

# Architecture

The paper proposes several network variants:

ResNet-18
ResNet-34
ResNet-50
ResNet-101
ResNet-152

Deeper versions use **bottleneck residual blocks** to reduce computational cost.

---

# Experimental Results

Dataset: ImageNet

ResNet achieved state-of-the-art performance:

ResNet-152
Top-5 error: 3.57%

The model won **1st place in the ImageNet 2015 classification challenge**.

---

# Transfer to Other Tasks

The learned features generalize well to other tasks.

Object detection experiment:

Framework: Faster R-CNN

Backbone comparison:

VGG-16 → ResNet-101

Result:

COCO mAP improvement: +6%

This demonstrates that ResNet produces significantly stronger visual representations.

---

# Key Contributions

1. Introduced residual learning framework
2. Solved degradation problem in deep networks
3. Enabled training of 100+ layer CNNs
4. Achieved state-of-the-art results on ImageNet
5. Became the backbone of modern computer vision models

---

# Repository Structure

paper/
Detailed study notes

implementation/
PyTorch implementation of ResNet

experiments/
Experiments comparing ResNet with plain CNN

figures/
Illustrations and diagrams from the paper

---

# References

He, K., Zhang, X., Ren, S., & Sun, J. (2016).
Deep Residual Learning for Image Recognition.
CVPR 2016.
