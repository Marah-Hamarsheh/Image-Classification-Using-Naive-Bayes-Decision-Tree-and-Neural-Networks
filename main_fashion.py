"""
Comparative Study of Image Classification Using Decision Tree,
Naive Bayes, and Feedforward Neural Networks

Done by: Joud Thaher, Marah Hamarsheh
Date: 22/6/2025
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.neural_network import MLPClassifier


def load_data():
    """Load and normalize preprocessed dataset"""
    data = np.load("fashion_mnist_data.npz")  # Changed filename here
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    # Normalize pixel values to [0,1] if not already done
    if X_train.max() > 1.0:
        X_train = X_train / 255.0
        X_test = X_test / 255.0

    return X_train, y_train, X_test, y_test


def plot_confusion_matrix(cm, classes, title):
    """Plot confusion matrix with annotations"""
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def evaluate_model(model, X_test, y_test, class_names, model_name=None):
    """Evaluate model and visualize results"""
    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='macro'),
        'recall': recall_score(y_test, y_pred, average='macro'),
        'f1': f1_score(y_test, y_pred, average='macro')
    }

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Print results
    model_display_name = model_name if model_name else model.__class__.__name__
    print(f"\n=== {model_display_name} Results ===")
    print("Evaluation Metrics:")
    for name, value in metrics.items():
        print(f"- {name.capitalize()}: {value:.4f}")

    # Plot confusion matrix
    plot_confusion_matrix(cm, class_names,
                         f"Confusion Matrix ({model_display_name})")
    plt.show()

    return metrics


def run_naive_bayes(X_train, y_train, X_test, y_test, class_names):
    """Optimized Naive Bayes implementation"""
    print("\n" + "="*70)
    print("RUNNING NAIVE BAYES CLASSIFIER")
    print("="*70)
    model = GaussianNB(var_smoothing=1e-9)  # Tuned smoothing parameter
    model.fit(X_train, y_train)
    return evaluate_model(model, X_test, y_test, class_names, "Naive Bayes")


def run_decision_tree(X_train, y_train, X_test, y_test, class_names):
    """Robust Decision Tree implementation with multiple visualization options"""
    print("\n" + "=" * 70)
    print("RUNNING DECISION TREE CLASSIFIER")
    print("=" * 70)

    # Create and fit the model
    model = DecisionTreeClassifier(
        max_depth=5,  # Reduced from 10 for better visualization
        min_samples_split=5,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test, class_names, "Decision Tree")

    try:
        plt.figure(figsize=(24, 12))
        plot_tree(model, max_depth=2, filled=True)
        plt.savefig('decision_tree.png', dpi=300, bbox_inches='tight')
        print("\nTree visualization saved to 'decision_tree.png'")
        plt.close()
    except Exception as e:
        print(f"Could not save tree image: {str(e)}")

    return metrics


def run_neural_network(X_train, y_train, X_test, y_test, class_names):
    """Optimized Neural Network implementation"""
    print("\n" + "="*70)
    print("RUNNING FEEDFORWARD NEURAL NETWORK")
    print("="*70)
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        max_iter=500,
        early_stopping=True,
        random_state=42
    )
    model.fit(X_train, y_train)
    return evaluate_model(model, X_test, y_test, class_names, "Neural Network")


def compare_models(results):
    """Generate comparative analysis of models"""
    print("\n" + "=" * 70)
    print("Comparative Analysis")
    print("=" * 70)
    print("{:<25} {:<10} {:<10} {:<10} {:<10}".format(
        'Model', 'Accuracy', 'Precision', 'Recall', 'F1-score'))

    for model_name, metrics in results.items():
        print("{:<25} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
            model_name,
            metrics['accuracy'],
            metrics['precision'],
            metrics['recall'],
            metrics['f1']
        ))


def main():
    # Configuration - Updated class names for Fashion-MNIST (3 classes only)
    CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Sandal']

    # Load data
    X_train, y_train, X_test, y_test = load_data()
    print(f"\nData loaded: {X_train.shape[0]} train samples, {X_test.shape[0]} test samples")

    # Run and evaluate classifiers
    results = {
        'Naive Bayes': run_naive_bayes(X_train, y_train, X_test, y_test, CLASS_NAMES),
        'Decision Tree': run_decision_tree(X_train, y_train, X_test, y_test, CLASS_NAMES),
        'Neural Network': run_neural_network(X_train, y_train, X_test, y_test, CLASS_NAMES)
    }

    # Comparative analysis
    compare_models(results)

    # Feature importance visualization (for Decision Tree)
    if 'Decision Tree' in results:
        try:
            plt.figure(figsize=(10, 5))
            plt.bar(range(len(CLASS_NAMES)), results['Decision Tree']['feature_importances_'][:len(CLASS_NAMES)])
            plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES)
            plt.title("Feature Importances (Decision Tree)")
            plt.show()
        except:
            pass


if __name__ == "__main__":
    main()
