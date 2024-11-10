import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

# Define transformations
transform_mnist_to_vector = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])

transform_cifar10_to_vector = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])

def load_complete_dataset():
    # Load MNIST dataset
    mnist_dataset = datasets.MNIST(
        root='./data', train=True, transform=transform_mnist_to_vector, download=True
    )
    
    # Load CIFAR-10 dataset
    cifar10_dataset = datasets.CIFAR10(
        root='./data', train=True, transform=transform_cifar10_to_vector, download=True
    )
    
    # Convert datasets to tensors
    mnist_data = torch.stack([img for img, _ in mnist_dataset])
    mnist_labels = torch.tensor([label for _, label in mnist_dataset])
    
    cifar_data = torch.stack([img for img, _ in cifar10_dataset])
    cifar_labels = torch.tensor([label for _, label in cifar10_dataset])
    
    print("Full MNIST data shape:", mnist_data.shape)
    print("Full CIFAR-10 data shape:", cifar_data.shape)
    
    return mnist_data, mnist_labels, cifar_data, cifar_labels

# For batch processing if needed
batch_size = 128
mnist_loader = DataLoader(datasets.MNIST(
    root='./data', train=True, transform=transform_mnist_to_vector, download=True
), batch_size=batch_size, shuffle=True)

cifar10_loader = DataLoader(datasets.CIFAR10(
    root='./data', train=True, transform=transform_cifar10_to_vector, download=True
), batch_size=batch_size, shuffle=True)
