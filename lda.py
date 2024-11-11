import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

__all__ = ['analyze_with_lda']

def standardize_data(data, device='cuda', batch_size=4000):
    """
    Standardize data using GPU acceleration and batch processing
    """
    # Reference implementation from pca.py
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
    
    return standardized_data, mean, std

def compute_class_means(data, labels, n_classes, device='cuda'):
    """
    Compute mean vectors for each class
    """
    n_features = data.shape[1]
    class_means = torch.zeros((n_classes, n_features), device=device)
    class_counts = torch.zeros(n_classes, device=device)
    
    for i in range(n_classes):
        mask = (labels == i)
        class_data = data[mask]
        class_means[i] = class_data.mean(dim=0)
        class_counts[i] = mask.sum()
    
    return class_means, class_counts

def compute_scatter_matrices(data, labels, class_means, class_counts, batch_size=1000, device='cuda'):
    """
    Compute within-class and between-class scatter matrices
    """
    n_classes = class_means.shape[0]
    n_features = data.shape[1]
    
    # Initialize scatter matrices
    S_w = torch.zeros((n_features, n_features), device=device)
    S_b = torch.zeros((n_features, n_features), device=device)
    
    # Compute global mean
    global_mean = torch.mean(data, dim=0)
    
    # Compute between-class scatter matrix
    for i in range(n_classes):
        diff = (class_means[i] - global_mean).unsqueeze(1)
        S_b += class_counts[i] * torch.matmul(diff, diff.t())
    
    # Compute within-class scatter matrix in batches
    n_samples = data.shape[0]
    n_batches = (n_samples - 1) // batch_size + 1
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        
        batch_data = data[start_idx:end_idx]
        batch_labels = labels[start_idx:end_idx]
        
        for j in range(n_classes):
            mask = (batch_labels == j)
            if mask.any():
                class_data = batch_data[mask]
                diff = class_data - class_means[j]
                S_w += torch.matmul(diff.t(), diff)
    
    return S_w, S_b

def analyze_with_lda(data, labels, n_components=None, batch_size=1000, device='cuda'):
    """
    Perform LDA dimension reduction using GPU acceleration
    
    Parameters:
    -----------
    data : array-like or torch.Tensor
        Input data matrix of shape (n_samples, n_features)
    labels : array-like or torch.Tensor
        Class labels of shape (n_samples,)
    n_components : int, optional
        Number of components to keep. If None, will keep min(n_classes-1, n_features)
    batch_size : int, default=1000
        Batch size for processing
    device : str, default='cuda'
        Device to use for computation ('cuda' or 'cpu')
    
    Returns:
    --------
    reduced_data : numpy.ndarray
        Transformed data matrix of shape (n_samples, n_components)
    """
    if not torch.cuda.is_available() and device == 'cuda':
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    # Convert inputs to torch tensors if needed
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)
    
    # Move data to device
    data = data.to(device)
    labels = labels.to(device)
    
    # Get dimensions
    n_samples, n_features = data.shape
    n_classes = len(torch.unique(labels))
    
    if n_components is None:
        n_components = min(n_classes - 1, n_features)
    
    print(f"\nPerforming LDA reduction to {n_components} components...")
    
    # Standardize data
    data, mean, std = standardize_data(data, device, batch_size)
    
    # Compute class means and counts
    class_means, class_counts = compute_class_means(data, labels, n_classes, device)
    
    # Compute scatter matrices
    S_w, S_b = compute_scatter_matrices(data, labels, class_means, class_counts, batch_size, device)
    
    # Solve generalized eigenvalue problem
    print("\nSolving generalized eigenvalue problem...")
    try:
        eigenvals, eigenvecs = torch.linalg.eigh(torch.matmul(torch.linalg.inv(S_w), S_b))
    except RuntimeError:
        # If S_w is singular, add small regularization
        print("Adding regularization to within-class scatter matrix...")
        S_w += torch.eye(n_features, device=device) * 1e-4
        eigenvals, eigenvecs = torch.linalg.eigh(torch.matmul(torch.linalg.inv(S_w), S_b))
    
    # Sort eigenvectors by eigenvalues in descending order
    idx = torch.argsort(eigenvals, descending=True)
    eigenvecs = eigenvecs[:, idx]
    
    # Select top k eigenvectors
    W = eigenvecs[:, :n_components]
    
    # Project data onto new space
    print("\nProjecting data...")
    reduced_data = torch.matmul(data, W)
    
    # Move result back to CPU
    reduced_data = reduced_data.cpu().numpy()
    print(f"Final data shape: {reduced_data.shape}")
    
    # Clear GPU memory
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    return reduced_data 