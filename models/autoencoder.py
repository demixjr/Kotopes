import torch
import torch.nn as nn


class SimpleAutoencoder(nn.Module):
    """
    Простой автоэнкодер: сжимает изображение и восстанавливает его
    """
    def __init__(self, latent_dim=128):
        super(SimpleAutoencoder, self).__init__()
        
        # Encoder - сжимает изображение
        self.encoder = nn.Sequential(
            # 3x224x224 -> 32x112x112
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            
            # 32x112x112 -> 64x56x56
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            
            # 64x56x56 -> 128x28x28
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            
            # 128x28x28 -> 256x14x14
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            
            # Flatten
            nn.Flatten(),
            
            # 256x14x14 -> latent_dim
            nn.Linear(256 * 14 * 14, latent_dim)
        )
        
        # Decoder - восстанавливает изображение
        self.decoder = nn.Sequential(
            # latent_dim -> 256x14x14
            nn.Linear(latent_dim, 256 * 14 * 14),
            nn.ReLU(),
            
            # Unflatten
            nn.Unflatten(1, (256, 14, 14)),
            
            # 256x14x14 -> 128x28x28
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            
            # 128x28x28 -> 64x56x56
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            
            # 64x56x56 -> 32x112x112
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            
            # 32x112x112 -> 3x224x224
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Значения от 0 до 1
        )
        
        self.latent_dim = latent_dim
    
    def encode(self, x):
        """Сжатие изображения в вектор"""
        return self.encoder(x)
    
    def decode(self, z):
        """Восстановление изображения из вектора"""
        return self.decoder(z)
    
    def forward(self, x):
        """Полный цикл: сжатие -> восстановление"""
        z = self.encode(x)
        reconstructed = self.decode(z)
        return reconstructed


class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) - может генерировать новые изображения
    """
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Латентное пространство: mu и logvar
        self.fc_mu = nn.Linear(256 * 14 * 14, latent_dim)
        self.fc_logvar = nn.Linear(256 * 14 * 14, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 256 * 14 * 14)
        
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, 14, 14)),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        
        self.latent_dim = latent_dim
    
    def encode(self, x):
        """Получаем mu и logvar"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Трюк репараметризации для sampling"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Восстановление изображения"""
        h = self.decoder_input(z)
        return self.decoder(h)
    
    def forward(self, x):
        """Полный forward pass"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar
    
    def sample(self, num_samples, device):
        """Генерация новых изображений"""
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decode(z)
        return samples


def vae_loss(reconstructed, original, mu, logvar):
    """
    Loss для VAE: reconstruction loss + KL divergence
    """
    # Reconstruction loss
    recon_loss = nn.functional.mse_loss(reconstructed, original, reduction='sum')
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + kl_loss


if __name__ == "__main__":
    # Тест автоэнкодера
    print("Testing Simple Autoencoder...")
    ae = SimpleAutoencoder(latent_dim=128)
    x = torch.randn(2, 3, 224, 224)
    reconstructed = ae(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {reconstructed.shape}")
    
    # Тест VAE
    print("\nTesting VAE...")
    vae = VAE(latent_dim=128)
    reconstructed, mu, logvar = vae(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {reconstructed.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"Logvar shape: {logvar.shape}")