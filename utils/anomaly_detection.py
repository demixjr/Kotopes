import torch
import numpy as np
import matplotlib.pyplot as plt


def calculate_reconstruction_error(model, dataloader, device):
    """
    Вычисляет ошибку реконструкции для каждого изображения
    Чем больше ошибка - тем более "странное" изображение
    """
    model.eval()
    
    errors = []
    all_images = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            
            # Реконструируем
            if hasattr(model, 'reparameterize'):  # VAE
                reconstructed, _, _ = model(images)
            else:  # Simple AE
                reconstructed = model(images)
            
            # Считаем ошибку для каждого изображения
            error = torch.mean((images - reconstructed) ** 2, dim=[1, 2, 3])
            
            errors.append(error.cpu().numpy())
            all_images.append(images.cpu().numpy())
            all_labels.append(labels.numpy())
    
    errors = np.concatenate(errors)
    all_images = np.concatenate(all_images)
    all_labels = np.concatenate(all_labels)
    
    return errors, all_images, all_labels


def detect_anomalies(errors, threshold_percentile=95):
    """
    Находит аномалии - изображения с большой ошибкой реконструкции
    """
    threshold = np.percentile(errors, threshold_percentile)
    anomalies = errors > threshold
    
    print(f"Threshold (percentile {threshold_percentile}): {threshold:.4f}")
    print(f"Found {np.sum(anomalies)} anomalies out of {len(errors)} images")
    
    return anomalies, threshold


def visualize_anomalies(errors, images, labels, num_normal=4, num_anomalies=4, save_path=None):
    """
    Показывает нормальные и аномальные изображения
    """
    # Сортируем по ошибке
    sorted_indices = np.argsort(errors)
    
    # Берем самые нормальные (маленькая ошибка)
    normal_indices = sorted_indices[:num_normal]
    
    # Берем самые странные (большая ошибка)
    anomaly_indices = sorted_indices[-num_anomalies:]
    
    # Визуализация
    fig, axes = plt.subplots(2, max(num_normal, num_anomalies), figsize=(12, 5))
    
    # Нормальные изображения
    for i in range(num_normal):
        idx = normal_indices[i]
        axes[0, i].imshow(np.transpose(images[idx], (1, 2, 0)))
        axes[0, i].set_title(f'Normal\nError: {errors[idx]:.4f}', fontsize=10)
        axes[0, i].axis('off')
    
    # Заполняем пустые места
    for i in range(num_normal, max(num_normal, num_anomalies)):
        axes[0, i].axis('off')
    
    # Аномальные изображения
    for i in range(num_anomalies):
        idx = anomaly_indices[i]
        axes[1, i].imshow(np.transpose(images[idx], (1, 2, 0)))
        axes[1, i].set_title(f'Anomaly\nError: {errors[idx]:.4f}', fontsize=10, color='red')
        axes[1, i].axis('off')
    
    # Заполняем пустые места
    for i in range(num_anomalies, max(num_normal, num_anomalies)):
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def plot_error_distribution(errors, labels, save_path=None):
    """
    Показывает распределение ошибок реконструкции
    """
    plt.figure(figsize=(12, 5))
    
    # Общая гистограмма
    plt.subplot(1, 2, 1)
    plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Reconstruction Errors')
    plt.grid(True, alpha=0.3)
    
    # Средний threshold (95 percentile)
    threshold = np.percentile(errors, 95)
    plt.axvline(threshold, color='red', linestyle='--', label=f'95% threshold: {threshold:.4f}')
    plt.legend()
    
    # По классам
    plt.subplot(1, 2, 2)
    unique_labels = np.unique(labels)
    colors = ['red', 'green', 'blue']
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.hist(errors[mask], bins=30, alpha=0.5, label=f'Class {label}', color=colors[i % len(colors)])
    
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution by Class')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    print("Anomaly detection utilities loaded!")
    print("Available functions:")
    print("  - calculate_reconstruction_error()")
    print("  - detect_anomalies()")
    print("  - visualize_anomalies()")
    print("  - plot_error_distribution()")