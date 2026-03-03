import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

def save_confusion_matrix(true_labels, predictions, class_names, save_path):
    """Save confusion matrix plot"""
    cm = confusion_matrix(true_labels, predictions)
    
    plt.figure(figsize=(20, 16))
    
    # Show only top N classes if too many
    if len(class_names) > 20:
        unique, counts = np.unique(true_labels, return_counts=True)
        top_classes = unique[np.argsort(counts)[-20:]]
        mask = np.isin(np.arange(len(class_names)), top_classes)
        cm_filtered = cm[mask][:, mask]
        class_names_filtered = [class_names[i] for i in top_classes]
    else:
        cm_filtered = cm
        class_names_filtered = class_names
    
    sns.heatmap(cm_filtered, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names_filtered,
                yticklabels=class_names_filtered)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return cm

def find_outliers(predictions, true_labels, confidences, 
                  image_paths, class_names, threshold=0.3):
    """Find potential outliers in predictions"""
    outliers = []
    
    for i in range(len(predictions)):
        if confidences[i] < threshold and predictions[i] != true_labels[i]:
            outliers.append({
                'index': i,
                'image_path': image_paths[i],
                'true_label': true_labels[i],
                'true_name': class_names.get(true_labels[i], f"Class_{true_labels[i]}"),
                'pred_label': predictions[i],
                'pred_name': class_names.get(predictions[i], f"Class_{predictions[i]}"),
                'confidence': confidences[i],
                'confidence_gap': threshold - confidences[i]
            })
    
    outliers.sort(key=lambda x: x['confidence_gap'], reverse=True)
    return outliers

def plot_training_history(history, save_path):
    """Plot training and validation loss"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training History')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot
    axes[1].plot(history['artist_acc'], label='Artist Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def save_outliers_report(outliers, save_path, top_k=50):
    """Save outliers report to file"""
    with open(save_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("OUTLIER DETECTION REPORT\n")
        f.write("="*80 + "\n\n")
        
        for i, outlier in enumerate(outliers[:top_k]):
            f.write(f"Outlier #{i+1}\n")
            f.write(f"  Image: {os.path.basename(outlier['image_path'])}\n")
            f.write(f"  True Artist: {outlier['true_name']}\n")
            f.write(f"  Predicted: {outlier['pred_name']}\n")
            f.write(f"  Confidence: {outlier['confidence']:.4f}\n")
            f.write(f"  Confidence Gap: {outlier['confidence_gap']:.4f}\n")
            f.write("-"*40 + "\n")