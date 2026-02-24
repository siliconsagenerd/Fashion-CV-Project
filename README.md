# Fashion-MNIST Classification: CNN vs. Random Forest

## Project Overview

This repository contains the code for the **DLBAIPCV01 – Computer Vision** project report (Task 2). It presents a comparative study between a traditional machine learning approach (Random Forest) and a deep learning approach (Convolutional Neural Network) for image classification on the Zalando Fashion-MNIST dataset.

The goal is to evaluate both classifiers, identify their strengths and weaknesses across the 10 fashion categories, and recommend which model is better suited for production use.

## Dataset

Fashion-MNIST (Xiao et al., 2017) is a drop-in replacement for the classic MNIST digit dataset. It consists of:

- **70,000** grayscale images at **28×28** pixels
- **60,000** training / **10,000** test split
- **10 classes:** T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
- Balanced distribution (6,000 samples per class in the training set)

## Model Architectures

### Random Forest Classifier
- 500 estimators (aligned with Xiao et al., 2017)
- Flattened 784-dimensional input vectors
- `max_features='sqrt'`, parallelised across all CPU cores

### Convolutional Neural Network
| Layer | Output Shape | Parameters |
|-------|-------------|------------|
| Conv2D (32 filters, 3×3) | 26 × 26 × 32 | 320 |
| MaxPooling2D (2×2) | 13 × 13 × 32 | 0 |
| Conv2D (64 filters, 3×3) | 11 × 11 × 64 | 18,496 |
| MaxPooling2D (2×2) | 5 × 5 × 64 | 0 |
| Flatten | 1,600 | 0 |
| Dense (128, ReLU) | 128 | 204,928 |
| Dense (10, Softmax) | 10 | 1,290 |

**Total trainable parameters:** 225,034

Trained for 10 epochs with Adam optimiser and sparse categorical cross-entropy loss.

## Key Results (Test Set)

| Metric | Random Forest | CNN |
|--------|--------------|-----|
| Accuracy | 87.86% | 91.40% |
| Macro Precision | 87.74% | 91.41% |
| Macro Recall | 87.86% | 91.40% |
| Macro F1 | 87.71% | 91.35% |
| Training Time | ~49 s | ~135 s |

The Random Forest achieves 100% training accuracy, indicating overfitting to the training data. The CNN generalises better, reaching ~97% on train and ~91% on test.

Both models struggle most with **Shirt**, which is visually similar to T-shirt/top, Pullover, and Coat in 28×28 grayscale.

## Generated Outputs

Running the script produces the following files in the `images/` folder:

| File | Description |
|------|-------------|
| `sample_images.png` | One sample image per class |
| `class_distribution.png` | Training set class balance |
| `training_curves.png` | CNN accuracy and loss over epochs |
| `rf_train_cm.png` | Random Forest training confusion matrix |
| `rf_test_cm.png` | Random Forest test confusion matrix |
| `cnn_train_cm.png` | CNN training confusion matrix |
| `cnn_test_cm.png` | CNN test confusion matrix |
| `cnn_misclassified.png` | Misclassified examples from the CNN's worst class |
| `per_class_metrics.csv` | Per-class precision, recall, and F1 for both models |
| `summary_metrics.csv` | Macro-average metrics and accuracy comparison |

## How to Run

```bash
# 1. Create a virtual environment (optional but recommended)
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the script
python fashion_main.py
```

All outputs are saved automatically to the `images/` folder.

## Project Structure

```
Fashion_CV_Project/
├── fashion_main.py        # Main script – trains both models and generates all outputs
├── requirements.txt       # Python dependencies
├── data/                  # Cached dataset (auto-downloaded on first run)
└── images/                # Generated figures and CSV tables
```

## References

- Xiao, H., Rasul, K., & Vollgraf, R. (2017). *Fashion-MNIST: a novel image dataset for benchmarking machine learning algorithms.* arXiv:1708.07747.
- TensorFlow (2024). *Basic classification: Classify images of clothing.* https://www.tensorflow.org/tutorials/keras/classification

