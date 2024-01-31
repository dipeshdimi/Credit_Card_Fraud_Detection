# Credit Card Fraud Detection

## Introduction
This repository contains a Jupyter Notebook (`Credit_Card_Fraud_Detection.ipynb`) that focuses on detecting fraudulent credit card transactions using logistic regression. The notebook includes steps for data loading, exploration, preprocessing, model training, and evaluation.

## Dataset
The dataset used in this project can be found on [Credit Card Fraud Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud). It consists of transactions labeled as legitimate (Class 0) or fraudulent (Class 1). Please note that the dataset is relatively large, so the initial loading may take some time.

## Dependencies
To run the notebook, make sure you have the following dependencies installed:
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
```
## Exploratory Data Analysis
The notebook includes an exploration of the dataset, covering aspects such as data overview, information, checking for missing values, and analyzing the class distribution.

## Data Balancing
Given the highly unbalanced nature of the dataset, with a significant number of legitimate transactions (Class 0) and fewer fraudulent transactions (Class 1), the notebook implements under-sampling to balance the dataset for training purposes.

## Model Training
Logistic Regression is chosen as the classification algorithm for this task. The notebook includes code for training the logistic regression model using the balanced dataset.

## Model Evaluation
The notebook evaluates the trained model on both the training and testing datasets, providing metrics such as accuracy, confusion matrix, precision, recall, and F1 score.

## Results
After training and evaluating the logistic regression model, the notebook displays the performance metrics on both the training and testing datasets.

