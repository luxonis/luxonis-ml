import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import Grayscale, Lambda

def load_mnist_data(save_path='./mnist', num_samples=640, batch_size=64):
    # Define the transformations for preprocessing the image data
    transform = transforms.Compose([
        Grayscale(num_output_channels=3),  # Convert images to grayscale
        Lambda(lambda x: x.convert("RGB")),  # Convert grayscale to RGB
        transforms.Resize((224, 224)),  # Resize images to (224, 224)
        transforms.ToTensor(),  # Convert images to tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
    ])

    # Load the MNIST dataset
    dataset = torchvision.datasets.MNIST(root=save_path, train=True, transform=transform, download=True)

    if num_samples == -1 :
        num_samples = len(dataset)

    # Define the indices to include in the subset (e.g., first 1000 samples)
    subset_indices = torch.arange(num_samples)

    # Create a subset of the dataset using Subset class
    subset = torch.utils.data.Subset(dataset, subset_indices)

    # Create a data loader to load the dataset in batches
    data_loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=4)

    return data_loader
