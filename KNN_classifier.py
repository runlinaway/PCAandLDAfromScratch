import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

def knn_comparison(original_data, reduced_data, labels, n_neighbors=5, test_size=0.2, random_state=42):
    """
    Compare KNN classification performance on original and PCA-reduced data
    
    Parameters:
    -----------
    original_data : array-like
        Original high-dimensional data
    reduced_data : array-like
        PCA-reduced data
    labels : array-like
        Target labels
    n_neighbors : int, default=5
        Number of neighbors for KNN
    test_size : float, default=0.2
        Proportion of dataset to include in the test split
    random_state : int, default=42
        Random state for reproducibility
    
    Returns:
    --------
    dict : Dictionary containing accuracy scores and classification reports
    """
    
    # Convert to numpy arrays if they're torch tensors
    if hasattr(original_data, 'numpy'):
        original_data = original_data.numpy()
    if hasattr(reduced_data, 'numpy'):
        reduced_data = reduced_data.numpy()
    if hasattr(labels, 'numpy'):
        labels = labels.numpy()
    
    # Split the data
    X_train_orig, X_test_orig, X_train_red, X_test_red, y_train, y_test = train_test_split(
        original_data, reduced_data, labels, test_size=test_size, random_state=random_state
    )
    
    # Initialize KNN classifiers
    knn_original = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_reduced = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    # Fit and predict on original data
    knn_original.fit(X_train_orig, y_train)
    y_pred_orig = knn_original.predict(X_test_orig)
    
    # Fit and predict on reduced data
    knn_reduced.fit(X_train_red, y_train)
    y_pred_red = knn_reduced.predict(X_test_red)
    
    # Calculate metrics
    results = {
        'original_accuracy': accuracy_score(y_test, y_pred_orig),
        'reduced_accuracy': accuracy_score(y_test, y_pred_red),
        'original_report': classification_report(y_test, y_pred_orig),
        'reduced_report': classification_report(y_test, y_pred_red)
    }
    
    return results 