# PCA Analysis on MNIST and CIFAR-10 Datasets

## Overview
This project implements Principal Component Analysis (PCA) on the MNIST and CIFAR-10 datasets using PyTorch with GPU acceleration. It analyzes the dimensionality reduction capabilities of PCA and compares classification performance before and after reduction using K-Nearest Neighbors (KNN) classification.

## Features
- **GPU-accelerated PCA implementation**
- **Batch processing** for memory efficiency
- **Scree plot** visualization for variance analysis
- **KNN classification comparison** before and after dimensionality reduction
- **Support for both MNIST and CIFAR-10 datasets**

## Requirements
- **Python 3.10+**
- **PyTorch**
- **NumPy**
- **Matplotlib**
- **scikit-learn**
- **CUDA-capable GPU** (optional but recommended)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/runlinaway/PCAandLDAfromScratch.git
   cd <repository-folder>
2. Install dependencies:
   ```bash
   pip install torch torchvision numpy matplotlib scikit-learn

