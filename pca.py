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

def plot_scree(data, dataset_name, n_components=None, batch_size=4000, device='cuda'):
    """
    Create a scree plot using GPU acceleration
    """
    if not torch.cuda.is_available() and device == 'cuda':
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    # Convert to torch tensor if needed
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()
    
    # Move data to device
    data = data.to(device)
    
    # Initialize parameters
    n_samples, n_features = data.shape
    batch_size = max(batch_size, n_features + 1)
    n_components = min(batch_size - 1, n_features) if n_components is None else min(n_components, batch_size - 1)
    
    print(f"\nAnalyzing {dataset_name} variance with {n_components} components...")
    print(f"Using batch size of {batch_size} on {device}")
    
    # Standardize data
    data = standardize_data(data, device, batch_size)
    
    # Compute covariance matrix in batches
    print("\nComputing covariance matrix...")
    cov_matrix = torch.zeros((n_features, n_features), device=device)
    
    n_batches = (n_samples - 1) // batch_size + 1
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        batch = data[start_idx:end_idx]
        
        cov_matrix += torch.matmul(batch.T, batch)
        
        if i % 10 == 0:
            print(f"Processed batch {i+1}/{n_batches}")
    
    cov_matrix /= (n_samples - 1)
    
    # Compute eigenvalues and eigenvectors
    print("\nComputing eigendecomposition...")
    eigenvals, eigenvecs = torch.linalg.eigh(cov_matrix)
    
    # Sort in descending order
    eigenvals = eigenvals.flip(0)
    eigenvecs = eigenvecs.flip(1)
    
    # Calculate variance ratios
    var_exp = (eigenvals / eigenvals.sum()).cpu().numpy()
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    components = range(1, len(var_exp) + 1)
    ax1.plot(components, var_exp, 'bo-')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title(f'{dataset_name} - Individual Components')
    
    cumsum = np.cumsum(var_exp)
    ax2.plot(components, cumsum, 'ro-')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance Ratio')
    ax2.set_title(f'{dataset_name} - Cumulative')
    
    thresholds = [0.7, 0.8, 0.9, 0.95]
    colors = ['g', 'y', 'orange', 'r']
    k_values = {}
    
    for threshold, color in zip(thresholds, colors):
        ax2.axhline(y=threshold, color=color, linestyle='--', alpha=0.5)
        k = next((i for i, x in enumerate(cumsum) if x >= threshold), len(cumsum))
        k_values[threshold] = k + 1
        ax2.text(len(components) * 0.02, threshold + 0.02, 
                f'{threshold*100}% variance at k={k + 1}', 
                color=color)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n{dataset_name} Variance Analysis:")
    for threshold, k in k_values.items():
        print(f"Components needed for {threshold*100}% variance: {k}")
    
    return var_exp, k_values, eigenvecs.cpu()

def analyze_with_pca(data, dataset_name, k, batch_size=1000, device='cuda'):
    """
    Perform PCA dimension reduction using GPU acceleration
    """
    if not torch.cuda.is_available() and device == 'cuda':
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    # Convert to torch tensor if needed
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()
    
    # Move data to device
    data = data.to(device)
    
    print(f"\nReducing {dataset_name} dimensions...")
    
    # Standardize data
    data = standardize_data(data, device, batch_size)
    
    # Compute covariance matrix in batches
    n_samples, n_features = data.shape
    print("\nComputing covariance matrix...")
    cov_matrix = torch.zeros((n_features, n_features), device=device)
    
    n_batches = (n_samples - 1) // batch_size + 1
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        batch = data[start_idx:end_idx]
        cov_matrix += torch.matmul(batch.T, batch)
        
    cov_matrix /= (n_samples - 1)
    
    # Compute eigenvalues and eigenvectors
    print("\nComputing eigendecomposition...")
    eigenvals, eigenvecs = torch.linalg.eigh(cov_matrix)
    
    # Sort in descending order
    eigenvecs = eigenvecs.flip(1)
    eigenvecs = eigenvecs.to(device)
    
    # Project data onto principal components
    n_samples = data.shape[0]
    reduced_data = torch.zeros((n_samples, k), device=device)
    
    n_batches = (n_samples - 1) // batch_size + 1
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        batch = data[start_idx:end_idx]
        
        # Project batch onto first k eigenvectors
        reduced_data[start_idx:end_idx] = torch.matmul(batch, eigenvecs[:, :k])
        
        if i % 10 == 0:
            print(f"Processed batch {i+1}/{n_batches}")
    
    # Move result back to CPU
    reduced_data = reduced_data.cpu().numpy()
    print(f"Reduced {dataset_name} shape: {reduced_data.shape}")
    
    # Clear GPU memory
    if device == 'cuda':
        torch.cuda.empty_cache()
    
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

