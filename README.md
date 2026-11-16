# CS231n Assignment 1: Image Classification

This repository contains the implementation of Assignment 1 from the Stanford CS231n course. The project explores the basic Image Classification pipeline, cross-validation, and vectorized code implementation.[cite: 5]

## Overview
The assignment involves building and training various image classification models from scratch on the **CIFAR-10 dataset**.[cite: 5] The implemented models and techniques include:

*   **k-Nearest Neighbor (kNN):** Building a kNN classifier, computing distance matrices (using vectorized operations), and tuning the hyperparameter `k` via cross-validation.[cite: 5]
*   **Multiclass Support Vector Machine (SVM):** Implementing a fully-vectorized SVM loss function and analytic gradient, and optimizing the loss with Stochastic Gradient Descent (SGD).[cite: 11]
*   **Softmax Classifier:** Implementing a fully-vectorized Softmax loss function and analytic gradient, tuned and optimized with SGD.[cite: 9]
*   **Image Features:** Extracting Histogram of Oriented Gradients (HOG) and color histograms (HSV) to improve classification performance compared to training on raw pixels.[cite: 3]
*   **Two-Layer Neural Network:** Training a two-layer neural network on both raw pixels and extracted image features to achieve higher classification accuracy.[cite: 3]

## Dataset
*   **CIFAR-10 Data:** Utilizes 50,000 training images and 10,000 test images.[cite: 5]
*   Images are preprocessed by flattening into rows, subtracting the mean image, and appending a bias dimension to optimize weight matrix calculations.[cite: 9]

## Repository Structure
*   `knn.ipynb`: k-Nearest Neighbor exercise.[cite: 5]
*   `svm.ipynb`: Multiclass SVM exercise.[cite: 11]
*   `softmax.ipynb`: Softmax Classifier exercise.[cite: 9]
*   `features.ipynb`: Image features and neural network evaluation.[cite: 3]
*   `requirements.txt`: Python package dependencies.[cite: 7]
*   `setup_googlecloud.sh`: Environment setup script for Google Cloud instances.

## Installation & Setup
A setup script is provided for initializing a virtual environment and installing the required packages specified in `requirements.txt`.[cite: 7, 8]

```bash
# Create and activate a virtual environment
virtualenv .env
source .env/bin/activate

# Install dependencies
pip install -r requirements.txt
