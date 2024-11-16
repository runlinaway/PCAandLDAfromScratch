import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

__all__ = ['analyze_with_pca', 'plot_scree', 'reconstruct_from_pca']

def standardize_data(data, device='cuda', batch_size=4000):
    """
    Standardize data using GPU acceleration and batch processing
    """
    n_samples, n_features = data.shape
    
    # Initialize statistics tensors on GPU
    mean = torch.zeros(n_features, device=device)
    var = torch.zeros(n_features, device=device)
    
    # Compute mean and variance in batches
    n_batches = (n_samples - 1) // batch_size + 1
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        batch = data[start_idx:end_idx].to(device)
        
        mean += batch.mean(dim=0)
        var += batch.var(dim=0, unbiased=False)
    
    mean /= n_batches
    var /= n_batches
    std = torch.sqrt(var + 1e-8)
    
    # Standardize data in batches
    standardized_data = torch.zeros_like(data, device=device)
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        batch = data[start_idx:end_idx].to(device)
        standardized_data[start_idx:end_idx] = (batch - mean) / std
    
    return standardized_data

def plot_scree(loader, dataset_name, n_components=None, batch_size=1000, device='cuda'):
    """
    Perform PCA and create scree plot for the given data
    
    Parameters:
    -----------
    loader : DataLoader
        DataLoader containing the dataset
    dataset_name : str
        Name of the dataset for plotting
    n_components : int, optional
        Number of components to compute
    batch_size : int
        Batch size for processing
    device : str
        Device to use for computation
    """
    # Get all data from the loader
    data = loader.dataset.data
    
    # Get data dimensions
    n_samples, n_features = data.shape
    
    if n_components is None:
        n_components = min(n_samples, n_features)
    
    # Center the data
    mean = torch.mean(data, dim=0)
    centered_data = data - mean
    
    # Compute covariance matrix
    print("Computing covariance matrix...")
    cov_matrix = torch.mm(centered_data.T, centered_data) / (n_samples - 1)
    
    # Compute eigenvalues and eigenvectors
    print("Computing eigendecomposition...")
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
    
    # Sort eigenvalues and eigenvectors in descending order
    idx = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Calculate explained variance ratio
    total_var = torch.sum(eigenvalues)
    explained_var_ratio = eigenvalues / total_var
    
    # Calculate cumulative explained variance ratio
    cumulative_var_ratio = torch.cumsum(explained_var_ratio, dim=0)
    
    # Find k values for different variance thresholds
    k_values = {}
    thresholds = [0.8, 0.85, 0.9, 0.95, 0.99]
    for threshold in thresholds:
        k = torch.where(cumulative_var_ratio >= threshold)[0][0].item() + 1
        k_values[threshold] = k
    
    # Create scree plot
    plt.figure(figsize=(10, 5))
    
    # Plot individual explained variance
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(explained_var_ratio) + 1), 
             explained_var_ratio.cpu().numpy(), 'bo-')
    plt.title(f'{dataset_name} Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    
    # Plot cumulative explained variance
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(cumulative_var_ratio) + 1), 
             cumulative_var_ratio.cpu().numpy(), 'ro-')
    plt.title(f'{dataset_name} Cumulative Explained Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    
    # Add horizontal lines for thresholds
    for threshold in thresholds:
        plt.axhline(y=threshold, color='gray', linestyle='--', alpha=0.5)
        plt.text(len(cumulative_var_ratio) * 0.6, threshold, 
                f'{threshold:.0%} - {k_values[threshold]} components', 
                verticalalignment='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return explained_var_ratio.cpu().numpy(), k_values, eigenvectors.cpu().numpy()

def analyze_with_pca(loader, dataset_name, k, batch_size=1000, device='cuda'):
    """
    Perform PCA dimension reduction
    
    Parameters:
    -----------
    loader : DataLoader
        DataLoader containing the dataset
    dataset_name : str
        Name of the dataset
    k : int
        Number of components to keep
    batch_size : int
        Batch size for processing
    device : str
        Device to use for computation
    """
    # Get all data from the loader
    data = loader.dataset.data
    
    # Center the data
    mean = torch.mean(data, dim=0)
    centered_data = data - mean
    
    # Compute covariance matrix
    n_samples = data.shape[0]
    cov_matrix = torch.mm(centered_data.T, centered_data) / (n_samples - 1)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
    
    # Sort eigenvalues and eigenvectors in descending order
    idx = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Select top k eigenvectors
    selected_eigenvectors = eigenvectors[:, :k]
    
    # Project data onto new space
    reduced_data = torch.mm(centered_data, selected_eigenvectors)
    
    return reduced_data

def reconstruct_from_pca(reduced_data, eigenvecs, mean=None, std=None, device='cuda'):
    """
    Reconstruct original data from PCA components using GPU acceleration
    """
    if not torch.cuda.is_available() and device == 'cuda':
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    # Convert to torch tensor if needed
    if isinstance(reduced_data, np.ndarray):
        reduced_data = torch.from_numpy(reduced_data).float()
    if isinstance(eigenvecs, np.ndarray):
        eigenvecs = torch.from_numpy(eigenvecs).float()
    
    # Move data to device
    reduced_data = reduced_data.to(device)
    eigenvecs = eigenvecs.to(device)
    
    # Reconstruct data
    reconstructed = torch.matmul(reduced_data, eigenvecs.T)
    
    # Unstandardize if mean and std are provided
    if mean is not None and std is not None:
        mean = mean.to(device)
        std = std.to(device)
        reconstructed = (reconstructed * std) + mean
    
    # Move result back to CPU
    reconstructed = reconstructed.cpu().numpy()
    
    # Clear GPU memory
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    return reconstructed

