import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np

class ImageDataset(Dataset):
    """Custom Dataset class for image data with normalization and centering"""
    def __init__(self, dataset_name, train=True, device='cuda'):
        self.device = device
        
        # Basic transform to convert to tensor
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        # Load the appropriate dataset
        if dataset_name == 'mnist':
            dataset = datasets.MNIST(root='./data', train=train, download=True)
        elif dataset_name == 'fashion_mnist':
            dataset = datasets.FashionMNIST(root='./data', train=train, download=True)
        elif dataset_name == 'cifar':
            dataset = datasets.CIFAR10(root='./data', train=train, download=True)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
            
        # Process the data
        self.data = []
        self.labels = []
        
        print(f"Loading {dataset_name} dataset...")
        for img, label in dataset:
            # Convert PIL image to tensor and flatten
            img_tensor = self.transform(img)
            img_flat = img_tensor.view(-1)
            
            self.data.append(img_flat)
            self.labels.append(label)
        
        # Convert to tensors
        self.data = torch.stack(self.data).to(device)
        self.labels = torch.tensor(self.labels).to(device)
        
        # Normalize and center the data
        print("Normalizing and centering data...")
        self.mean = torch.mean(self.data, dim=0)
        self.std = torch.std(self.data, dim=0)
        
        # Handle constant features (std = 0)
        self.std[self.std == 0] = 1
        
        # Center and normalize
        self.data = (self.data - self.mean) / self.std
        
        print(f"Loaded {len(self.data)} samples")
        print(f"Data shape: {self.data.shape}")
        print(f"Mean: {torch.mean(self.data):.4f}, Std: {torch.std(self.data):.4f}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    def get_statistics(self):
        """Return dataset statistics"""
        return {
            'mean': self.mean,
            'std': self.std,
            'data_mean': torch.mean(self.data),
            'data_std': torch.std(self.data),
            'data_min': torch.min(self.data),
            'data_max': torch.max(self.data)
        }

def create_data_loaders(batch_size=128, device='cuda'):
    """Create DataLoaders for all datasets"""
    # Create datasets
    datasets_dict = {
        'mnist': ImageDataset('mnist', device=device),
        'fashion_mnist': ImageDataset('fashion_mnist', device=device),
        'cifar': ImageDataset('cifar', device=device)
    }
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    for name, dataset in datasets_dict.items():
        stats = dataset.get_statistics()
        print(f"\n{name}:")
        print(f"Overall Mean: {stats['data_mean']:.4f}")
        print(f"Overall Std: {stats['data_std']:.4f}")
        print(f"Min Value: {stats['data_min']:.4f}")
        print(f"Max Value: {stats['data_max']:.4f}")
    
    # DataLoader settings
    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': 0,  # Avoid multiprocessing
        'pin_memory': False  # Data is already on GPU
    }
    
    # Create loaders
    loaders = {}
    for name, dataset in datasets_dict.items():
        loaders[name] = DataLoader(dataset, **loader_kwargs)
        
    return loaders

# Example usage
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loaders = create_data_loaders(batch_size=128, device=device)
    
    # Print dataset sizes
    for name, loader in loaders.items():
        data_shape = next(iter(loader))[0].shape
        print(f"{name} batch shape: {data_shape}")


