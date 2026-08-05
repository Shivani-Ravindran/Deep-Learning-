# Experiment 3 – Convolutional Neural Network for Multi-Class Image Classification

## Objective

Design, implement, train and evaluate a Convolutional Neural Network (CNN) using TensorFlow/Keras for multi-class image classification on the CIFAR-10 dataset. Investigate the influence of hyperparameters (optimizer, batch size) on training, diagnose overfitting by comparing training/validation curves with and without data augmentation, and compare Max Pooling against Average Pooling as the final architectural choice.

## Dataset

**CIFAR-10 Dataset** (TensorFlow/Keras)

| Property | Value |
|---|---|
| Training Images | 50,000 |
| Testing Images | 10,000 |
| Classes | 10 |
| Image Size | 32 × 32 (RGB, 3 channels) |
| Missing Values | None |

## Notebook Structure

| Cell(s) | Task | Description |
|---|---|---|
| 0 | Setup | Configures matplotlib plotting style |
| 1 | Imports | Loads TensorFlow/Keras, seaborn, matplotlib and scikit-learn metrics |
| 2 | Dataset Loading | Downloads CIFAR-10 using `tensorflow.keras.datasets` |
| 3 | Dataset Inspection | Prints tensor shapes of train/test splits |
| 4 | Class Distribution | Flattens labels and plots the training set class distribution |
| 5 | Sample Images | Displays ten randomly sampled training images with their class labels |
| 6 | Preprocessing | Normalizes pixel values to [0, 1] and one-hot encodes the class labels |
| 7 | Hyperparameter Exploration | Trains a baseline Max Pooling CNN (no augmentation) for 5 epochs under four optimizer/batch-size configurations and compares validation accuracy/loss |
| 8 | Overfitting Check (Max Pooling) | Trains the Max Pooling CNN for 10 epochs without augmentation and plots training/validation accuracy and loss to check for overfitting |
| 9–10 | Final Model (Max Pooling + Augmentation) | Adds a data augmentation block (random flip, rotation, zoom), builds and trains the final Max Pooling CNN for 10 epochs |
| 11 | Training Curves | Plots training/validation accuracy and loss for the augmented Max Pooling model |
| 12 | Model Evaluation | Predicts on the test set and computes accuracy, precision, recall, F1-score, classification report and confusion matrix |
| 13 | Overfitting Check (Average Pooling) | Trains the Average Pooling CNN for 10 epochs without augmentation, for comparison against the Max Pooling baseline |
| 14–15 | Pooling Comparison (Average Pooling + Augmentation) | Builds and trains the augmented Average Pooling CNN and plots its training/validation accuracy and loss curves |

## How to Run

1. Open the notebook in Google Colab or Jupyter Notebook.
2. Run all cells sequentially from top to bottom.
3. CIFAR-10 is downloaded automatically from TensorFlow during execution.
4. The hyperparameter exploration step (cell 7) trains four separate models and is the most time-consuming step besides the initial data load.
5. All plots are displayed inline, and evaluation metrics are printed during execution.

## Key Implementation Notes

- The **CNN architecture** used throughout is `32×32×3 → Conv2D(32,3×3,ReLU) → Pool(2×2) → Conv2D(64,3×3,ReLU) → Pool(2×2) → Conv2D(128,3×3,ReLU) → Pool(2×2) → Flatten → Dense(128,ReLU) → Dropout(0.5) → Dense(10,Softmax)`.
- **Two pooling variants** are compared: `MaxPooling2D` and `AveragePooling2D`, with all other layers, optimizer, batch size and epochs held identical.
- **Data augmentation** (random horizontal flip, rotation, zoom) is applied ahead of the convolutional stack for the final models, added after the no-augmentation runs showed a widening train/validation gap indicating overfitting.
- **Hyperparameter exploration** compares Adam vs. SGD at batch sizes 32 and 128, using 5-epoch training runs on the unaugmented Max Pooling baseline.
- Pixel values are normalized to `[0, 1]` and labels are one-hot encoded prior to training.

## Results Summary

### Overfitting Check – No Augmentation (10 epochs)

| Pooling | Final Train Acc. | Final Val Acc. | Final Train Loss | Final Val Loss |
|---|---|---|---|---|
| Max Pooling | 0.7968 | 0.7520 | 0.5801 | 0.7546 |
| Average Pooling | 0.7292 | 0.7504 | 0.7784 | 0.7419 |

### Test-Set Performance (Max Pooling Model, with Augmentation)

| Metric | Value |
|---|---|
| Accuracy | 0.6702 |
| Precision (macro) | 0.6845 |
| Recall (macro) | 0.6702 |
| F1-score (macro) | 0.6596 |

### Pooling Strategy Comparison (Final Epoch, with Augmentation)

| Pooling | Train Acc. | Val Acc. | Train Loss | Val Loss |
|---|---|---|---|---|
| Max Pooling | 0.6218 | 0.6776 | 1.0815 | 0.9239 |
| Average Pooling | 0.6112 | 0.6910 | 1.1066 | 0.8710 |

### Additional Notes

Training loss fell steadily across all 10 epochs in both no-augmentation runs, while validation loss plateaued and fluctuated from around epoch 6–7 onward — the classic signature of overfitting even though the final-epoch accuracy gap looked modest. Introducing data augmentation removed this pattern in the subsequent runs, at the cost of slower convergence per epoch. Once augmentation was applied, Average Pooling generalised marginally better than Max Pooling (higher validation accuracy, lower validation loss).