import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.autoencoder import SimpleAutoencoder, VAE, vae_loss


def train_autoencoder(model, dataloader, optimizer, device, is_vae=False):
    """
    Обучение автоэнкодера на одну эпоху
    """
    model.train()
    total_loss = 0
    
    for images, _ in dataloader:
        images = images.to(device)
        
        optimizer.zero_grad()
        
        if is_vae:
            # VAE
            reconstructed, mu, logvar = model(images)
            loss = vae_loss(reconstructed, images, mu, logvar)
        else:
            # Простой AE
            reconstructed = model(images)
            loss = nn.functional.mse_loss(reconstructed, images, reduction='sum')
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss


def validate_autoencoder(model, dataloader, device, is_vae=False):
    """
    Валидация автоэнкодера
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            
            if is_vae:
                reconstructed, mu, logvar = model(images)
                loss = vae_loss(reconstructed, images, mu, logvar)
            else:
                reconstructed = model(images)
                loss = nn.functional.mse_loss(reconstructed, images, reduction='sum')
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss


def main():
    # Параметры
    DATA_DIR = 'data'
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LATENT_DIM = 128
    LEARNING_RATE = 0.001
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')
    
    # Трансформации (без нормализации для автоэнкодера)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # Загрузка данных
    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Train: {len(train_dataset)}")
    print(f"Val: {len(val_dataset)}")
    
    # ===== Обучение Simple Autoencoder =====
    print("\n" + "="*60)
    print("TRAINING SIMPLE AUTOENCODER")
    print("="*60)
    
    ae = SimpleAutoencoder(latent_dim=LATENT_DIM).to(device)
    optimizer_ae = optim.Adam(ae.parameters(), lr=LEARNING_RATE)
    
    ae_history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(NUM_EPOCHS):
        train_loss = train_autoencoder(ae, train_loader, optimizer_ae, device, is_vae=False)
        val_loss = validate_autoencoder(ae, val_loader, device, is_vae=False)
        
        ae_history['train_loss'].append(train_loss)
        ae_history['val_loss'].append(val_loss)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Сохранение модели
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(ae.state_dict(), 'checkpoints/autoencoder_best.pth')
    print("\n✓ Saved autoencoder to checkpoints/autoencoder_best.pth")
    
    # ===== Обучение VAE =====
    print("\n" + "="*60)
    print("TRAINING VAE")
    print("="*60)
    
    vae = VAE(latent_dim=LATENT_DIM).to(device)
    optimizer_vae = optim.Adam(vae.parameters(), lr=LEARNING_RATE)
    
    vae_history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(NUM_EPOCHS):
        train_loss = train_autoencoder(vae, train_loader, optimizer_vae, device, is_vae=True)
        val_loss = validate_autoencoder(vae, val_loader, device, is_vae=True)
        
        vae_history['train_loss'].append(train_loss)
        vae_history['val_loss'].append(val_loss)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Сохранение модели
    torch.save(vae.state_dict(), 'checkpoints/vae_best.pth')
    print("\n✓ Saved VAE to checkpoints/vae_best.pth")
    
    # ===== Визуализация =====
    os.makedirs('results/autoencoder', exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # AE Loss
    ax1.plot(ae_history['train_loss'], label='Train')
    ax1.plot(ae_history['val_loss'], label='Val')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Simple Autoencoder Loss')
    ax1.legend()
    ax1.grid(True)
    
    # VAE Loss
    ax2.plot(vae_history['train_loss'], label='Train')
    ax2.plot(vae_history['val_loss'], label='Val')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('VAE Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/autoencoder/training_curves.png')
    print("\n✓ Saved training curves to results/autoencoder/training_curves.png")


if __name__ == '__main__':
    main()