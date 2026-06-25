# NUS-Arrhythmia-detector

## Introduction
Cardiac arrhythmias are irregular heartbeats that can be life-threatening if undetected. Manual ECG interpretation is time-consuming and error-prone, making automation essential. This project presents a deep learning model combining CNN, BiLSTM, and a Transformer block to classify ECG signals from the MIT-BIH Arrhythmia Dataset. The architecture captures spatial, temporal, and global attention patterns, improving heartbeat classification accuracy for potential real-time healthcare applications.

## Problem Statement
Early and accurate detection of cardiac arrhythmias from ECG signals is critical for preventing serious health complications. Traditional diagnostic methods are often time-intensive and prone to human error. The challenge lies in developing an automated, robust, and accurate model that can effectively classify various types of heartbeats from raw ECG data. This project aims to build a deep learning-based solution—integrating CNN, BiLSTM, and Transformer layers—to enhance the classification performance on the MIT-BIH Arrhythmia Dataset.

## Dataset Overview
### Source and Purpose
The dataset is derived from the MIT-BIH Arrhythmia Database, a benchmark resource in medical research used for analyzing ECG (electrocardiogram) signals to detect and classify different types of heartbeats and arrhythmias.

### Structure and Format
It comprises two main CSV files: mitbih_train.csv and mitbih_test.csv. Each row represents one heartbeat, sampled over 187 time steps (features), separated by approx.. 2.7ms; with the final column indicating the class label (0–4) for heartbeat type.

### Heartbeat Classes
There are 5 distinct classes representing types of beats:
Class 0: Normal beats
Class 1: Supraventricular ectopic beats
Class 2: Ventricular ectopic beats
Class 3: Fusion beats
Class 4: Unknown beats

### Preprocessing & Reshaping: 
Data reshaped to (samples, 187, 1) for compatibility with 1D CNNs/LSTMs. Labels converted to integers for sparse_categorical_crossentropy.

### Class Imbalance: 
Dataset is skewed toward normal beats (class 0), requiring class weighting, oversampling, or advanced loss functions to handle minority classes.

### Exploratory Data Analysis
The Exploratory Data Analysis (EDA) phase involves analyzing and visualizing the dataset to understand its structure, distribution, and underlying patterns, which will help in building an effective deep learning model.
Here are the key steps involved in our EDA:

1.Data Loading:
Loaded the training and test datasets which contain ECG signal data and labels (heartbeat classes). The data consists of 5 classes.

2.Class Distribution:
Checked class distribution in training and test sets to identify imbalance. Visualizing this helps decide if class weighting or oversampling is needed.

3.Data Visualization:
Plot ECG signals from each class (Normal, Supraventricular, Ventricular, etc.) using line plots to compare heartbeat patterns and identify distinguishing features. 
Use histograms to visualize signal value distributions and detect outliers or anomalies.

### Methodology
#### 1) Data Preprocessing
The dataset (MIT-BIH) is pre-split into training and test sets.
Each sample contains 187 ECG signal values and a corresponding heartbeat label (0–4).
The data is reshaped into (samples, 187, 1) format for compatibility with Conv1D and LSTM layers.
Labels are converted into integer form for efficient processing using sparse categorical cross-entropy.

#### 2) Model Architecture
The model is a multi-stage hybrid pipeline combining three advanced techniques:

➤ Convolutional Layer (CNN)
Applies a 1D convolution with 64 filters and kernel size 5 to extract local patterns.
A MaxPooling layer follows to downsample and retain important features while reducing computational load.


➤ Bidirectional LSTM
Captures long-term dependencies in ECG data from both past and future directions using 64 LSTM units.
Ensures no important heartbeat pattern is missed from either end of the signal.

➤ Transformer Encoder Block
Integrates multi-head self-attention (4 heads) to identify critical time points in the signal.
Reduces each signal to a 128-dimensional vector via learned attention, enhancing model focus.
Includes layer normalization, dropout, and residual connections for stable and regularized learning.

#### 3) Classification Layer
GlobalAveragePooling1D aggregates the information from all time steps into a single vector.

This is passed through a Dense layer with 64 neurons and ReLU activation.

The final output layer uses softmax activation over 5 classes:
0: Normal
1: Supraventricular
2: Ventricular
3: Unknown
4: Fusion

#### 4) Model Compilation and Training
Optimizer: Adam — efficient for adaptive learning rates.

Loss Function: Sparse Categorical Crossentropy — suitable for integer-labeled multi-class problems.

Training Setup:
Epochs: 10
Batch Size: 128

Validation: Model accuracy and loss are tracked on the test set after every epoch to avoid overfitting.










