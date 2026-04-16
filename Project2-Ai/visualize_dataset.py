"""
Fashion-MNIST Dataset Visualization Script
This script loads and displays sample images from your prepared dataset
"""

import numpy as np
import matplotlib.pyplot as plt


def visualize_dataset():
    """Load and display sample images from the dataset"""

    # Load your prepared dataset
    print("Loading dataset...")
    data = np.load("fashion_mnist_data.npz")
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    # Class names
    class_names = ['T-shirt/top', 'Trouser', 'Sandal']

    print(f"Dataset info:")
    print(f"- Training samples: {X_train.shape[0]}")
    print(f"- Test samples: {X_test.shape[0]}")
    print(f"- Image features: {X_train.shape[1]} (28x28 flattened)")
    print(f"- Classes: {class_names}")

    # Reshape images back to 28x28 for visualization
    X_train_images = X_train.reshape(-1, 28, 28)
    X_test_images = X_test.reshape(-1, 28, 28)

    # Show training samples
    plt.figure(figsize=(15, 8))
    plt.suptitle('Training Dataset Samples', fontsize=16)

    samples_per_class = 8
    for class_idx in range(3):  # 3 classes
        # Find indices for this class
        class_indices = np.where(y_train == class_idx)[0]

        # Select random samples from this class
        selected_indices = np.random.choice(class_indices, samples_per_class, replace=False)

        for i, idx in enumerate(selected_indices):
            plt.subplot(3, samples_per_class, class_idx * samples_per_class + i + 1)
            plt.imshow(X_train_images[idx], cmap='gray')
            plt.title(f'{class_names[class_idx]}')
            plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Show test samples
    plt.figure(figsize=(15, 8))
    plt.suptitle('Test Dataset Samples', fontsize=16)

    for class_idx in range(3):  # 3 classes
        # Find indices for this class
        class_indices = np.where(y_test == class_idx)[0]

        # Select random samples from this class
        selected_indices = np.random.choice(class_indices, samples_per_class, replace=False)

        for i, idx in enumerate(selected_indices):
            plt.subplot(3, samples_per_class, class_idx * samples_per_class + i + 1)
            plt.imshow(X_test_images[idx], cmap='gray')
            plt.title(f'{class_names[class_idx]}')
            plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Show dataset statistics
    plt.figure(figsize=(12, 4))

    # Class distribution in training set
    plt.subplot(1, 3, 1)
    train_counts = np.bincount(y_train)
    plt.bar(range(3), train_counts, color=['skyblue', 'lightgreen', 'lightcoral'])
    plt.title('Training Set Distribution')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.xticks(range(3), class_names, rotation=45)

    # Class distribution in test set
    plt.subplot(1, 3, 2)
    test_counts = np.bincount(y_test)
    plt.bar(range(3), test_counts, color=['skyblue', 'lightgreen', 'lightcoral'])
    plt.title('Test Set Distribution')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.xticks(range(3), class_names, rotation=45)

    # Pixel intensity distribution
    plt.subplot(1, 3, 3)
    plt.hist(X_train.flatten(), bins=50, alpha=0.7, color='purple')
    plt.title('Pixel Intensity Distribution')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    # Show average image for each class
    plt.figure(figsize=(12, 4))
    plt.suptitle('Average Image per Class', fontsize=16)

    for class_idx in range(3):
        # Find all images of this class
        class_mask = y_train == class_idx
        class_images = X_train_images[class_mask]

        # Calculate average image
        avg_image = np.mean(class_images, axis=0)

        plt.subplot(1, 3, class_idx + 1)
        plt.imshow(avg_image, cmap='gray')
        plt.title(f'Average {class_names[class_idx]}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 50)
    print("Dataset visualization complete!")
    print("You can see:")
    print("1. Random samples from each class")
    print("2. Class distribution (should be balanced)")
    print("3. Pixel intensity distribution")
    print("4. Average image for each class")
    print("=" * 50)


if __name__ == "__main__":
    visualize_dataset()