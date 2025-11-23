import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def visualize_reconstructions(model, dataloader, device, num_images=8, save_path=None):
    """
    Показывает оригинальные изображения и их реконструкции
    """
    model.eval()
    
    # Берем батч изображений
    images, labels = next(iter(dataloader))
    images = images[:num_images].to(device)
    
    with torch.no_grad():
        if hasattr(model, 'reparameterize'):  # VAE
            reconstructed, _, _ = model(images)
        else:  # Simple AE
            reconstructed = model(images)
    
    # Переводим в numpy
    images = images.cpu().numpy()
    reconstructed = reconstructed.cpu().numpy()
    
    # Визуализация
    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 4))
    
    for i in range(num_images):
        # Оригинал
        axes[0, i].imshow(np.transpose(images[i], (1, 2, 0)))
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=12)
        
        # Реконструкция
        axes[1, i].imshow(np.transpose(reconstructed[i], (1, 2, 0)))
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def visualize_latent_space(model, dataloader, device, save_path=None):
    """
    Визуализация латентного пространства с помощью t-SNE
    """
    model.eval()
    
    latent_vectors = []
    labels_list = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            
            # Получаем латентные векторы
            if hasattr(model, 'reparameterize'):  # VAE
                mu, _ = model.encode(images)
                latent = mu
            else:  # Simple AE
                latent = model.encode(images)
            
            latent_vectors.append(latent.cpu().numpy())
            labels_list.append(labels.numpy())
    
    # Объединяем
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    labels_list = np.concatenate(labels_list, axis=0)
    
    # t-SNE для визуализации (если больше 2D)
    if latent_vectors.shape[1] > 2:
        print("Applying t-SNE...")
        tsne = TSNE(n_components=2, random_state=42)
        latent_2d = tsne.fit_transform(latent_vectors)
    else:
        latent_2d = latent_vectors
    
    # Визуализация
    plt.figure(figsize=(10, 8))
    
    unique_labels = np.unique(labels_list)
    colors = ['red', 'green', 'blue', 'orange', 'purple']
    
    for i, label in enumerate(unique_labels):
        mask = labels_list == label
        plt.scatter(latent_2d[mask, 0], latent_2d[mask, 1], 
                   c=colors[i % len(colors)], label=f'Class {label}', 
                   alpha=0.6, s=20)
    
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Latent Space Visualization (t-SNE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def visualize_generated_images(vae, device, num_images=16, save_path=None):
    """
    Генерация новых изображений с помощью VAE
    """
    vae.eval()
    
    with torch.no_grad():
        generated = vae.sample(num_images, device)
    
    generated = generated.cpu().numpy()
    
    # Визуализация
    rows = 4
    cols = num_images // rows
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    
    for i in range(num_images):
        row = i // cols
        col = i % cols
        axes[row, col].imshow(np.transpose(generated[i], (1, 2, 0)))
        axes[row, col].axis('off')
    
    plt.suptitle('Generated Images by VAE', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def visualize_denoising(model, dataloader, device, noise_level=0.3, num_images=8, save_path=None):
    """
    Демонстрация деноизинга (очистки от шума)
    """
    model.eval()
    
    # Берем изображения
    images, _ = next(iter(dataloader))
    images = images[:num_images].to(device)
    
    # Добавляем шум
    noisy_images = images + noise_level * torch.randn_like(images)
    noisy_images = torch.clamp(noisy_images, 0, 1)
    
    # Очищаем
    with torch.no_grad():
        if hasattr(model, 'reparameterize'):  # VAE
            denoised, _, _ = model(noisy_images)
        else:  # Simple AE
            denoised = model(noisy_images)
    
    # Переводим в numpy
    images = images.cpu().numpy()
    noisy_images = noisy_images.cpu().numpy()
    denoised = denoised.cpu().numpy()
    
    # Визуализация
    fig, axes = plt.subplots(3, num_images, figsize=(num_images * 2, 6))
    
    for i in range(num_images):
        # Оригинал
        axes[0, i].imshow(np.transpose(images[i], (1, 2, 0)))
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=12)
        
        # С шумом
        axes[1, i].imshow(np.transpose(noisy_images[i], (1, 2, 0)))
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Noisy', fontsize=12)
        
        # Очищенное
        axes[2, i].imshow(np.transpose(denoised[i], (1, 2, 0)))
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_title('Denoised', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    print("Visualization utilities loaded!")
    print("Available functions:")
    print("  - visualize_reconstructions()")
    print("  - visualize_latent_space()")
    print("  - visualize_generated_images()")
    print("  - visualize_denoising()")