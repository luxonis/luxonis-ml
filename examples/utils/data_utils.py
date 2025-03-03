"""MNIST Dataset Loader with Custom Preprocessing.

This module provides utility functions to load the MNIST dataset with specific
transformations applied. The dataset is transformed to be compatible with models
that expect 3-channel RGB images, such as models pre-trained on the ImageNet dataset.

The main transformations applied are:
    1. Convert single-channel grayscale images to 3-channel grayscale.
    2. Convert 3-channel grayscale images to RGB format (though the image will still look grayscale visually).
    3. Resize images to the size of (224, 224) to be compatible with models like ResNet-50.
    4. Convert images to PyTorch tensors.
    5. Normalize the tensor values using the mean and standard deviation of ImageNet.

Functions:
    - `mnist_transformations()`: Returns the composed transformations.
    - `load_mnist_data(save_path='./mnist', num_samples=640, batch_size=64)`: Loads MNIST data with the defined transformations
        and returns a DataLoader. It provides options to specify the number of samples and batch size.

Example usage:
    ```
    data_loader = load_mnist_data('./mnist_data', num_samples=1000, batch_size=32)
    for batch in data_loader:
        images, labels = batch
        ...  # Your processing here
    ```

Note: This loader is particularly useful when you want to use MNIST data with models that were
pre-trained on datasets like ImageNet and expect 3-channel RGB input.
"""

import torch
import torchvision
from torchvision import transforms


def mnist_transformations() -> transforms.Compose:
    """Returns composed transformations for the MNIST dataset.

    Transforms the images from 1 channel grayscale to 3 channels RGB and
    resizes them.
    """
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )


def load_mnist_data(
    save_path: str = "./mnist", num_samples: int = 640, batch_size: int = 64
) -> torch.utils.data.DataLoader:
    """Loads the MNIST dataset with the specified preprocessing.

    Parameters:
    - save_path (str): Directory to save/load the MNIST data.
    - num_samples (int): Number of samples to load from the dataset.
                         Set as -1 to load the entire dataset.
    - batch_size (int): Batch size for the DataLoader.

    Returns:
    - data_loader (DataLoader): DataLoader for the MNIST dataset.
    """
    transform = mnist_transformations()

    # Load the MNIST dataset
    dataset = torchvision.datasets.MNIST(
        root=save_path, train=True, transform=transform, download=True
    )

    # If num_samples is set to -1, use the entire dataset
    num_samples = (
        min(num_samples, len(dataset)) if num_samples != -1 else len(dataset)
    )

    # Create a subset of the dataset using Subset class
    subset = torch.utils.data.Subset(dataset, torch.arange(num_samples))

    # Create a data loader to load the dataset in batches
    data_loader = torch.utils.data.DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    return data_loader
