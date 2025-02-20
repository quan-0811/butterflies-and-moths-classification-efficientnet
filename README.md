# 100 Butterfly Species Classification with EfficientNet

## Overview
This project implements **EfficientNet**, specifically its baseline model - **EfficientNet_B0**, from PyTorch basic and fundamental building blocks. The goal is to classify **100 different butterfly species** based on their images, leveraging the **EfficientNet architecture** to achieve state-of-the-art performance with high accuracy and efficiency.

This replication mainly based on: **[EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks (Tan & Le, 2019)](https://arxiv.org/abs/1905.11946)**, the original paper

## Dataset
The dataset used for this project is **[Butterfly & Moths Image Classification 100 species](https://www.kaggle.com/datasets/gpiosenka/butterfly-images40-species)** dataset from Kaggle. It consists of 100 butterfly species in **224 x 224 x 3 jpg** format.

## Data Augmentation
In order to avoid overfitting, the training data is enriched with slightly modified original images, resulting in **4x times bigger** training dataset (1 original image, 3 augmented ones).

You can access `train.py` for the augmentation details.

## Results
The replicated EfficientNet_B0 achieved roughly **92% accuracy** on the training data and **87% accuracy** on test data, signaling good generalization capabilities.

## Deployment
A **Gradio-based web interface** is also provided by running `app.py`, allowing to upload an image of a butterfly and receive a prediction with confidence scores.



