# Image Classification with ML Classifiers

Comparing **Naive Bayes**, **Decision Tree**, and **Neural Network (MLP)** on a 3-class subset of the Fashion-MNIST dataset.

> Done by: Joud Thaher · Marah Hamarsheh — June 2025

---

## Project Structure

```
├── fashion_mnist_prep.py    # Download & prepare the dataset
├── main_fashion.py          # Train & evaluate all three models
├── visualize_dataset.py     # Visualize dataset samples & stats
├── decision_tree.png        # Saved decision tree plot
└── docs/
    ├── project_description.pdf  # Original assignment brief
    └── project_report.pdf       # Full report & findings
```

---

## Setup

```bash
pip install numpy matplotlib scikit-learn tensorflow
```

---

## How to Run

**1. Prepare the dataset**
```bash
python fashion_mnist_prep.py
```
Downloads Fashion-MNIST, selects 3 classes, and saves `fashion_mnist_data.npz` (500 train / 300 test samples).

**2. (Optional) Visualize the data**
```bash
python visualize_dataset.py
```

**3. Train & compare all models**
```bash
python main_fashion.py
```

---

## Models

| Model | Configuration |
|---|---|
| Naive Bayes | GaussianNB, `var_smoothing=1e-9` |
| Decision Tree | `max_depth=5`, `min_samples_split=5` |
| Neural Network | MLP (128→64), ReLU, Adam, early stopping |

---

## Dataset

- **Source:** Fashion-MNIST via `tf.keras.datasets`
- **Classes:** T-shirt/top · Trouser · Sandal
- **Split:** 500 training samples · 300 test samples (balanced)
- **Input:** 28×28 grayscale → 784 flattened features, normalized to [0, 1]

---

## Evaluation

Each model reports **accuracy, precision, recall, and F1-score** (macro-averaged) plus a confusion matrix.
