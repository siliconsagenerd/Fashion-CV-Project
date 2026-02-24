"""
Fashion-MNIST Classification: CNN vs Random Forest
Course: DLBAIPCV01 - Project: Computer Vision (Task 2)
"""

import ssl
import certifi
import os

# fix macOS SSL cert issue
os.environ['SSL_CERT_FILE'] = certifi.where()
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             precision_recall_fscore_support)
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf  # noqa: F401
import keras
from keras import layers

os.makedirs('images', exist_ok=True)

CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# ------------------------------------------------------------------
# 1. Load and explore the dataset
# ------------------------------------------------------------------
print("Loading Fashion-MNIST dataset...")

(train_images, train_labels), (test_images, test_labels) = \
    keras.datasets.fashion_mnist.load_data()

print(f"Training set: {train_images.shape[0]} images ({train_images.shape[1]}x{train_images.shape[2]})")
print(f"Test set:     {test_images.shape[0]} images")
print(f"Pixel range:  {train_images.min()} – {train_images.max()}")
print(f"Classes:      {len(CLASS_NAMES)}")

# show how many samples per class
print("\nSamples per class (train):")
for i, name in enumerate(CLASS_NAMES):
    print(f"  {name}: {np.sum(train_labels == i)}")

# save a grid showing one example from each class
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle('Fashion-MNIST - One Sample per Class', fontsize=14, fontweight='bold')
for idx, ax in enumerate(axes.flat):
    sample_idx = np.where(train_labels == idx)[0][0]
    ax.imshow(train_images[sample_idx], cmap='gray')
    ax.set_title(CLASS_NAMES[idx], fontsize=10)
    ax.axis('off')
plt.tight_layout()
plt.savefig('images/sample_images.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved sample_images.png")

# save class distribution bar chart
fig, ax = plt.subplots(figsize=(10, 4))
unique, counts = np.unique(train_labels, return_counts=True)
ax.bar([CLASS_NAMES[i] for i in unique], counts, color='steelblue')
ax.set_ylabel('Number of Samples')
ax.set_title('Training Set - Class Distribution')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('images/class_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved class_distribution.png")

# normalise pixels to 0-1
train_images = train_images / 255.0
test_images = test_images / 255.0


# ------------------------------------------------------------------
# 2. Train Random Forest
# ------------------------------------------------------------------
print("\nTraining Random Forest (500 trees)...")

# flatten images to 1D vectors for the tree model
X_train_flat = train_images.reshape(train_images.shape[0], -1)
X_test_flat = test_images.reshape(test_images.shape[0], -1)

rf_model = RandomForestClassifier(
    n_estimators=500,
    max_features='sqrt',
    n_jobs=-1,
    random_state=42
)

start = time.time()
rf_model.fit(X_train_flat, train_labels)
rf_train_time = time.time() - start
print(f"Done in {rf_train_time:.1f}s")

rf_train_preds = rf_model.predict(X_train_flat)
rf_test_preds = rf_model.predict(X_test_flat)


# ------------------------------------------------------------------
# 3. Train CNN
# ------------------------------------------------------------------
print("\nTraining CNN...")

# reshape to (N, 28, 28, 1) so Conv2D can use the spatial dimensions
X_train_cnn = train_images.reshape(-1, 28, 28, 1)
X_test_cnn = test_images.reshape(-1, 28, 28, 1)

cnn_model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

cnn_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

cnn_model.summary()

start = time.time()
history = cnn_model.fit(
    X_train_cnn, train_labels,
    epochs=10,
    batch_size=32,
    validation_data=(X_test_cnn, test_labels),
    verbose=1
)
cnn_train_time = time.time() - start
print(f"Done in {cnn_train_time:.1f}s")

cnn_train_preds = np.argmax(cnn_model.predict(X_train_cnn, verbose=0), axis=-1)
cnn_test_preds = np.argmax(cnn_model.predict(X_test_cnn, verbose=0), axis=-1)


# ------------------------------------------------------------------
# 4. Save training curves
# ------------------------------------------------------------------
print("\nSaving training curves...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

ax1.plot(history.history['accuracy'], label='Train')
ax1.plot(history.history['val_accuracy'], label='Validation')
ax1.set_title('CNN Accuracy per Epoch')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(history.history['loss'], label='Train')
ax2.plot(history.history['val_loss'], label='Validation')
ax2.set_title('CNN Loss per Epoch')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/training_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved training_curves.png")


# ------------------------------------------------------------------
# 5. Confusion matrices (train + test for both models)
# ------------------------------------------------------------------
print("\nGenerating confusion matrices...")


def save_confusion_matrix(y_true, y_pred, filename, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(title)
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.savefig(f'images/{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()


save_confusion_matrix(train_labels, rf_train_preds, 'rf_train_cm',
                      'Random Forest - Training Confusion Matrix')
save_confusion_matrix(test_labels, rf_test_preds, 'rf_test_cm',
                      'Random Forest - Test Confusion Matrix')
save_confusion_matrix(train_labels, cnn_train_preds, 'cnn_train_cm',
                      'CNN - Training Confusion Matrix')
save_confusion_matrix(test_labels, cnn_test_preds, 'cnn_test_cm',
                      'CNN - Test Confusion Matrix')
print("Saved all 4 confusion matrices.")


# ------------------------------------------------------------------
# 6. Per-class precision, recall, F1
# ------------------------------------------------------------------
print("\nCalculating per-class metrics...")


def per_class_metrics(y_true, y_pred, model_name, set_name):
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(10))
    rows = []
    for i in range(10):
        rows.append({
            'Model': model_name,
            'Set': set_name,
            'Class': CLASS_NAMES[i],
            'Precision': round(prec[i], 4),
            'Recall': round(rec[i], 4),
            'F1-Score': round(f1[i], 4),
            'Support': int(support[i])
        })
    return pd.DataFrame(rows)


metrics_df = pd.concat([
    per_class_metrics(train_labels, rf_train_preds, 'Random Forest', 'Train'),
    per_class_metrics(test_labels, rf_test_preds, 'Random Forest', 'Test'),
    per_class_metrics(train_labels, cnn_train_preds, 'CNN', 'Train'),
    per_class_metrics(test_labels, cnn_test_preds, 'CNN', 'Test'),
], ignore_index=True)

metrics_df.to_csv('images/per_class_metrics.csv', index=False)
print("Saved per_class_metrics.csv")

# print test-set tables
rf_test_df = metrics_df[(metrics_df['Model'] == 'Random Forest') &
                        (metrics_df['Set'] == 'Test')]
cnn_test_df = metrics_df[(metrics_df['Model'] == 'CNN') &
                         (metrics_df['Set'] == 'Test')]

print("\nRandom Forest (Test):")
print(rf_test_df[['Class', 'Precision', 'Recall', 'F1-Score']].to_string(index=False))

print("\nCNN (Test):")
print(cnn_test_df[['Class', 'Precision', 'Recall', 'F1-Score']].to_string(index=False))


# ------------------------------------------------------------------
# 7. Summary table + training times
# ------------------------------------------------------------------
print("\nSummary comparison:")


def macro_summary(y_true, y_pred, model_name, set_name):
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    return {
        'Model': model_name, 'Set': set_name,
        'Accuracy': round(accuracy_score(y_true, y_pred), 4),
        'Macro Precision': round(prec, 4),
        'Macro Recall': round(rec, 4),
        'Macro F1': round(f1, 4)
    }


summary_df = pd.DataFrame([
    macro_summary(train_labels, rf_train_preds, 'Random Forest', 'Train'),
    macro_summary(test_labels, rf_test_preds, 'Random Forest', 'Test'),
    macro_summary(train_labels, cnn_train_preds, 'CNN', 'Train'),
    macro_summary(test_labels, cnn_test_preds, 'CNN', 'Test'),
])
summary_df.to_csv('images/summary_metrics.csv', index=False)
print(summary_df.to_string(index=False))

print(f"\nTraining times:")
print(f"  Random Forest: {rf_train_time:.1f}s")
print(f"  CNN:           {cnn_train_time:.1f}s")


# ------------------------------------------------------------------
# 8. Worst-category analysis
# ------------------------------------------------------------------
print("\nWorst-performing classes (by F1, test set):")

for model_name, test_df in [('Random Forest', rf_test_df), ('CNN', cnn_test_df)]:
    worst = test_df.sort_values('F1-Score').head(3)
    print(f"\n  {model_name}:")
    for _, row in worst.iterrows():
        print(f"    {row['Class']}: P={row['Precision']:.3f}  R={row['Recall']:.3f}  F1={row['F1-Score']:.3f}")

# save a grid of misclassified images for the CNN's worst class
worst_class_idx = cnn_test_df.sort_values('F1-Score').iloc[0]
worst_label = CLASS_NAMES.index(worst_class_idx['Class'])
mask = (test_labels == worst_label) & (cnn_test_preds != worst_label)
wrong_indices = np.where(mask)[0][:10]

if len(wrong_indices) > 0:
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle(f'CNN Misclassified "{CLASS_NAMES[worst_label]}" Examples',
                 fontsize=13, fontweight='bold')
    for i, ax in enumerate(axes.flat):
        if i < len(wrong_indices):
            idx = wrong_indices[i]
            ax.imshow(test_images[idx].reshape(28, 28), cmap='gray')
            ax.set_title(f'Pred: {CLASS_NAMES[cnn_test_preds[idx]]}',
                         fontsize=9, color='red')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('images/cnn_misclassified.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved cnn_misclassified.png")


# ------------------------------------------------------------------
# 9. Quick comparison for reference
# ------------------------------------------------------------------
rf_test_acc = accuracy_score(test_labels, rf_test_preds)
cnn_test_acc = accuracy_score(test_labels, cnn_test_preds)

print(f"\nFinal test accuracy:")
print(f"  Random Forest: {rf_test_acc:.4f}")
print(f"  CNN:           {cnn_test_acc:.4f}")

print("\nDone! All images and CSVs saved to the images/ folder.")
