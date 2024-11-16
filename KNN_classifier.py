import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import torch
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.metrics import roc_curve, auc, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_classification(y_true, y_pred, y_pred_proba=None, class_names=None):
    """
    Comprehensive evaluation of classification results
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_pred_proba : array-like, optional
        Predicted probabilities for ROC curve
    class_names : list, optional
        Names of classes for plotting
    """
    metrics = {}
    
    # Basic metrics
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1_score'] = f1
    
    # Per-class metrics
    class_report = classification_report(y_true, y_pred, output_dict=True)
    metrics['class_report'] = class_report
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    if class_names:
        plt.xticks(np.arange(len(class_names)) + 0.5, class_names, rotation=45)
        plt.yticks(np.arange(len(class_names)) + 0.5, class_names, rotation=0)
    plt.tight_layout()
    plt.show()
    
    # ROC curve and AUC (if probabilities are provided)
    if y_pred_proba is not None:
        plt.figure(figsize=(10, 8))
        for i in range(len(class_names)):
            fpr, tpr, _ = roc_curve(y_true == i, y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc="lower right")
        plt.show()
    
    return metrics

def knn_comparison(original_data, reduced_data, labels, n_neighbors=5, test_size=0.2, random_state=42):
    """Modified KNN comparison with comprehensive metrics"""
    # Convert to numpy if needed
    if torch.is_tensor(original_data):
        original_data = original_data.cpu().numpy()
    if torch.is_tensor(reduced_data):
        reduced_data = reduced_data.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    
    # Split data
    X_train_orig, X_test_orig, X_train_red, X_test_red, y_train, y_test = train_test_split(
        original_data, reduced_data, labels, test_size=test_size, random_state=random_state
    )
    
    # Train and evaluate original data
    knn_orig = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_orig.fit(X_train_orig, y_train)
    y_pred_orig = knn_orig.predict(X_test_orig)
    y_pred_proba_orig = knn_orig.predict_proba(X_test_orig)
    
    # Train and evaluate reduced data
    knn_red = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_red.fit(X_train_red, y_train)
    y_pred_red = knn_red.predict(X_test_red)
    y_pred_proba_red = knn_red.predict_proba(X_test_red)
    
    # Get class names (assuming sequential class labels)
    class_names = [str(i) for i in range(len(np.unique(labels)))]
    
    # Evaluate both models
    print("\nOriginal Data Metrics:")
    orig_metrics = evaluate_classification(y_test, y_pred_orig, y_pred_proba_orig, class_names)
    
    print("\nReduced Data Metrics:")
    red_metrics = evaluate_classification(y_test, y_pred_red, y_pred_proba_red, class_names)
    
    return {
        'original_metrics': orig_metrics,
        'reduced_metrics': red_metrics,
        'original_accuracy': accuracy_score(y_test, y_pred_orig),
        'reduced_accuracy': accuracy_score(y_test, y_pred_red)
    } 