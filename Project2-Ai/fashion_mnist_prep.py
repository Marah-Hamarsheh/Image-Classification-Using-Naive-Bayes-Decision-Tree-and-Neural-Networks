
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


def prepare_fashion_mnist():
    """Download and prepare Fashion-MNIST dataset with 3 classes and 500+ images"""
    print("Downloading Fashion-MNIST dataset...")

    # Load the dataset using TensorFlow/Keras
    (x_train_full, y_train_full), (x_test_full, y_test_full) = tf.keras.datasets.fashion_mnist.load_data()

    print(f"Original shapes - Train: {x_train_full.shape}, Test: {x_test_full.shape}")

    # Select 3 visually distinct classes as per project requirements
    # 0: T-shirt/top, 1: Trouser, 5: Sandal
    selected_classes = [0, 1, 5]  # T-shirt, Trouser, Sandal
    class_names = ['T-shirt/top', 'Trouser', 'Sandal']

    print(f"Selected classes: {selected_classes}")
    print(f"Class names: {class_names}")

    # Filter training data for selected classes
    train_mask = np.isin(y_train_full, selected_classes)
    x_train_filtered = x_train_full[train_mask]
    y_train_filtered = y_train_full[train_mask]

    # Filter test data for selected classes
    test_mask = np.isin(y_test_full, selected_classes)
    x_test_filtered = x_test_full[test_mask]
    y_test_filtered = y_test_full[test_mask]

    # Remap labels to 0, 1, 2 (instead of 0, 1, 5)
    label_mapping = {selected_classes[i]: i for i in range(len(selected_classes))}
    y_train_filtered = np.array([label_mapping[label] for label in y_train_filtered])
    y_test_filtered = np.array([label_mapping[label] for label in y_test_filtered])

    # Take exactly 500 training samples (as per project requirements)
    # Stratified sampling to ensure equal representation
    x_train, _, y_train, _ = train_test_split(
        x_train_filtered, y_train_filtered,
        train_size=500,  # Exactly 500 samples as required
        stratify=y_train_filtered,
        random_state=42
    )

    # Use a reasonable subset of test data
    x_test, _, y_test, _ = train_test_split(
        x_test_filtered, y_test_filtered,
        train_size=300,  # 300 test samples (100 per class)
        stratify=y_test_filtered,
        random_state=42
    )

    print(f"Filtered shapes - Train: {x_train.shape}, Test: {x_test.shape}")

    # Flatten the images from (28, 28) to (784,) as your code expects
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    # Convert to float32 and normalize to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    print(f"Final shapes - Train: {x_train.shape}, Test: {x_test.shape}")
    print(f"Classes: {np.unique(y_train)}")
    print(f"Samples per class in training: {np.bincount(y_train)}")
    print(f"Samples per class in testing: {np.bincount(y_test)}")

    # Save in the format your main.py expects
    np.savez('fashion_mnist_data.npz',
             X_train=x_train, y_train=y_train,
             X_test=x_test, y_test=y_test)

    print("✅ Dataset saved as 'fashion_mnist_data.npz'")
    print("✅ Ready to use with your main.py code!")
    print(f"✅ Dataset contains exactly {len(x_train)} training samples")
    print(f"✅ Dataset contains {len(selected_classes)} classes as required")

    # Print class names for reference
    print("\nClass mapping:")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")

    return class_names


if __name__ == "__main__":
    class_names = prepare_fashion_mnist()