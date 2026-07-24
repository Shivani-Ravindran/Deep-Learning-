# Experiment 1 – Single Layer Perceptron for Binary Classification

## Objective

Implement a Single Layer Perceptron from scratch to perform binary classification on the Banknote Authentication dataset, understand the perceptron learning rule, visualize the learning process (error, weight, and bias evolution), and evaluate the classifier using standard metrics. Additionally, the perceptron is applied to AND, OR, and NOT logic gates to demonstrate convergence on linearly separable problems.

## Dataset

**Banknote Authentication Dataset** (UCI Machine Learning Repository)

| Property | Value |
|---|---|
| Instances | 1372 |
| Features | 4 numerical (variance, skewness, curtosis, entropy) |
| Classes | 2 (0: Authentic, 1: Forged) |
| Missing values | None |

## Notebook Structure

| Cell(s) | Task | Description |
|---|---|---|
| 0–1 | Setup | Uploads dataset file |
| 2–3 | Dataset Exploration | Loads CSV, prints head, shape, null counts, descriptive statistics |
| 4–7 | Exploratory Data Analysis | Feature histograms, correlation heatmap, pairwise scatter plots (colored by class), boxplots |
| 8 | Preprocessing | 80/20 train-test split, feature standardization via `StandardScaler` |
| 9 | Perceptron Implementation | `Perceptron` class — zero-initialized weights/bias, step activation, perceptron learning rule, per-epoch logging |
| 10 | Training & Evaluation | Fits on scaled training data, plots training error vs. epoch, computes confusion matrix (manual TP/TN/FP/FN) and accuracy/precision/recall/F1, plots weight evolution and bias evolution over epochs |
| 11 | Learning Rate Comparison | Trains separate perceptrons at η = 0.001, 0.01, 0.1, 0.5 and overlays their error-vs-epoch curves |
| 12–13 | Additional Task: Logic Gates | Second `Perceptron` class variant (per-**update**, not per-epoch, weight logging; early stopping on convergence) trains AND, OR, and NOT gates and plots the decision boundary after every weight update |

## How to Run

1. Open the notebook in Google Colab.
2. Run Cell 0 — it will prompt a file upload; upload `data_banknote_authentication.txt`
3. Run all remaining cells top to bottom.
4. All plots render inline; printed metrics and per-epoch logs appear in cell output.

## Key Implementation Notes

- **Two separate `Perceptron` classes are used** in this notebook, by design, since they serve different purposes:
  - The **banknote classifier** (Cell 9) logs weights/errors once per **epoch**, matching the "Epoch-wise Learning" performance table required for the main experiment.
  - The **logic-gate classifier** (Cell 12) logs weights once per **individual weight update** (i.e., every misclassification correction), and stops early once an epoch produces zero misclassifications — this matches the additional-task requirement to show weights "after each update" and plot the boundary's evolution.
- **Confusion matrix and metrics** for the banknote task are computed manually (direct TP/TN/FP/FN counting) rather than via `sklearn.metrics`, per the from-scratch spirit of the experiment.
- **NOT gate** is single-input, so its decision boundary is a threshold point on a 1D number line rather than a 2D separating line; the `plot_boundary_history` method handles this case separately from the 2D AND/OR case.

## Results Summary

| Metric | Value |
|---|---|
| Learning rate (η) | 0.01 |
| Epochs | 100 |
| Final weights | [-0.23796763 -0.26246497 -0.24866506 -0.01923165] |
| Final bias | -0.12999999999999998 |
| Accuracy | 0.9818181818181818 |
| Precision | 1.0 |
| Recall | 0.9606299212598425 |
| F1-score | 0.9799196787148594 |

