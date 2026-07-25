# Experiment 2 – Multi-Layer Perceptron for Multi-Class Image Classification

## Objective

Implement a Multi-Layer Perceptron (MLP) using TensorFlow/Keras for multi-class classification on the Fashion-MNIST dataset, perform image preprocessing, train and evaluate a baseline model, optimize its hyperparameters using `RandomizedSearchCV` with the SciKeras wrapper, and compare the optimized model against the baseline. Additionally, implement the Perceptron Learning Algorithm from scratch for the XOR gate, visualize the decision boundary after every weight update, and analyze why the algorithm fails to converge.

## Dataset

**Fashion-MNIST Dataset** (TensorFlow/Keras)

| Property | Value |
|---|---|
| Training Images | 60,000 |
| Testing Images | 10,000 |
| Features | 784 (28 × 28 grayscale images flattened) |
| Classes | 10 |
| Image Size | 28 × 28 |
| Missing Values | None |

## Notebook Structure

| Cell(s) | Task | Description |
|---|---|---|
| 0 | Setup | Imports libraries and configures plotting style |
| 1 | Dataset Loading | Downloads the Fashion-MNIST dataset using `tensorflow.keras.datasets` |
| 2 | Dataset Exploration | Displays ten sample Fashion-MNIST images and plots class distribution |
| 3 | Data Preprocessing | Prints dataset dimensions, flattens images, normalizes pixel values to [0,1], performs one-hot encoding, and prints tensor shapes before and after preprocessing |
| 4 | Baseline MLP Construction | Builds the baseline architecture `784 → Dense(128, ReLU) → Dense(64, ReLU) → Dense(10, Softmax)` and displays `model.summary()` |
| 5 | Model Training | Compiles the model using Adam and categorical cross-entropy, then trains for 20 epochs with batch size 32 and validation split |
| 6–7 | Model Evaluation | Predicts on the test set, computes accuracy, precision, recall, F1-score, classification report and confusion matrix, and plots training/validation accuracy and loss curves |
| 8–10 | Hyperparameter Optimization | Defines the SciKeras model, performs `RandomizedSearchCV` (5-fold cross-validation), retrains the best model, evaluates it on the test set, and compares baseline and optimized performance |
| 11 | Performance Comparison | Prints side-by-side comparison of baseline and optimized model metrics |
| 12–13 | Additional Task: XOR Perceptron | Implements a perceptron from scratch for the XOR gate, logs weights after every update, plots the decision boundary after each update, tracks misclassifications, and demonstrates non-convergence |
| 14–15 | Additional Analysis | Trains a two-layer sigmoid neural network on XOR, plots loss and weight evolution, and compares its convergence behaviour against the single-layer perceptron |

## How to Run

1. Open the notebook in Google Colab or Jupyter Notebook.
2. Run all cells sequentially from top to bottom.
3. Fashion-MNIST is downloaded automatically from TensorFlow during execution.
4. The Randomized Search (20 candidate models × 5-fold cross-validation) is the most time-consuming step.
5. All plots are displayed inline, and evaluation metrics are printed during execution.

## Key Implementation Notes

- The **baseline MLP** uses two hidden layers with ReLU activation (`128` and `64` neurons) followed by a Softmax output layer for 10-class classification.
- Images are **flattened from 28 × 28 to 784-dimensional vectors**, normalized by dividing pixel values by `255`, and labels are converted to one-hot encoded vectors.
- **RandomizedSearchCV** is used with the **SciKeras wrapper** to automatically search over:
  - Number of hidden layers
  - Hidden neurons
  - Learning rate
  - Batch size
  - Number of epochs
  - Optimizer
  - Activation function
  - Dropout rate
- The **XOR task** implements the perceptron learning rule entirely from scratch, recording every weight update and plotting the decision boundary after each update to illustrate why the perceptron cannot solve non-linearly separable problems.
- A **two-layer sigmoid neural network** is also trained on the XOR dataset to demonstrate how adding a hidden layer enables successful convergence.

## Results Summary

| Metric | Baseline Model | Optimized Model |
|---|---|---|
| Accuracy | 0.8832 | 0.8832 |
| Precision | 0.8835 | 0.8835 |
| Recall | 0.8832 | 0.8832 |
| F1-score | 0.8814 | 0.8814 |

### Best Hyperparameters

| Hyperparameter | Value |
|---|---|
| Hidden Layers | 1 |
| Hidden Neurons | 128 |
| Activation | Tanh |
| Optimizer | SGD |
| Learning Rate | 0.1 |
| Batch Size | 32 |
| Epochs | 30 |
| Dropout | 0.0 |
| Cross-validation Accuracy | 0.8645 |

### Additional Task

The XOR perceptron does **not converge** because the XOR problem is **not linearly separable**. Instead, the weights repeatedly cycle through the same sequence of values, causing the decision boundary to oscillate indefinitely. In contrast, the two-layer sigmoid neural network successfully learns the XOR mapping and converges to a stable solution.