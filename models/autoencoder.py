import torch
import torch.nn as nn


class SimpleAutoencoder(nn.Module):
    """
    Простий автоенкодер для зображень 64x64
    """
    def __init__(self, latent_dim=64):
        super(SimpleAutoencoder, self).__init__()
        
        # Encoder - стискає зображення 64x64
        self.encoder = nn.Sequential(
            # 3x64x64 -> 32x32x32
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            # 32x32x32 -> 64x16x16
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            # 64x16x16 -> 128x8x8
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            # Flatten
            nn.Flatten(),
            
            # 128x8x8 -> latent_dim
            nn.Linear(128 * 8 * 8, latent_dim)
        )
        
        # Decoder - відновлює зображення
        self.decoder = nn.Sequential(
            # latent_dim -> 128x8x8
            nn.Linear(latent_dim, 128 * 8 * 8),
            nn.ReLU(),
            
            # Unflatten
            nn.Unflatten(1, (128, 8, 8)),
            
            # 128x8x8 -> 64x16x16
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            # 64x16x16 -> 32x32x32
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            # 32x32x32 -> 3x64x64
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Значення від 0 до 1
        )
        
        self.latent_dim = latent_dim
    
    def encode(self, x):
        """Стиснення зображення у вектор"""
        return self.encoder(x)
    
    def decode(self, z):
        """Відновлення зображення з вектора"""
        return self.decoder(z)
    
    def forward(self, x):
        """Повний цикл: стиснення -> відновлення"""
        z = self.encode(x)
        reconstructed = self.decode(z)
        return reconstructed


class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) - може генерувати нові зображення
    """
    def __init__(self, latent_dim=64):
        super(VAE, self).__init__()
        
        # Encoder для 64x64
        self.encoder = nn.Sequential(
            # 3x64x64 -> 32x32x32
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            # 32x32x32 -> 64x16x16
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            # 64x16x16 -> 128x8x8
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.Flatten()
        )
        
        # Латентний простір: mu і logvar
        self.encoder_output_size = 128 * 8 * 8  # Розмір після encoder
        self.fc_mu = nn.Linear(self.encoder_output_size, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_output_size, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, self.encoder_output_size)
        
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 8, 8)),
            
            # 128x8x8 -> 64x16x16
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            # 64x16x16 -> 32x32x32
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            # 32x32x32 -> 3x64x64
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        
        self.latent_dim = latent_dim
    
    def encode(self, x):
        """Отримуємо mu і logvar"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Трюк репараметризації для sampling"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Відновлення зображення"""
        h = self.decoder_input(z)
        return self.decoder(h)
    
    def forward(self, x):
        """Повний forward pass"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar
    
    def sample(self, num_samples, device):
        """Генерація нових зображень"""
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decode(z)
        return samples


# Спрощена версія для дуже швидкого навчання
class FastAutoencoder(nn.Module):
    """
    Дуже простий автоенкодер для максимально швидкого навчання
    """
    def __init__(self, latent_dim=32):
        super(FastAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # 3x64x64 -> 16x32x32
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            
            # 16x32x32 -> 32x16x16
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32 * 16 * 16),
            nn.ReLU(),
            nn.Unflatten(1, (32, 16, 16)),
            
            # 32x16x16 -> 16x32x32
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            
            # 16x32x32 -> 3x64x64
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


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
    # Тест автоенкодерів
    print("Testing Autoencoders for 64x64 images...")
    
    # Тест SimpleAutoencoder
    print("\n1. Testing Simple Autoencoder...")
    ae = SimpleAutoencoder(latent_dim=64)
    x = torch.randn(2, 3, 64, 64)
    reconstructed = ae(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {reconstructed.shape}")
    
    # Тест VAE
    print("\n2. Testing VAE...")
    vae = VAE(latent_dim=64)
    reconstructed, mu, logvar = vae(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {reconstructed.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"Logvar shape: {logvar.shape}")
    
    # Тест FastAutoencoder
    print("\n3. Testing Fast Autoencoder...")
    fast_ae = FastAutoencoder(latent_dim=32)
    reconstructed_fast = fast_ae(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {reconstructed_fast.shape}")
    
    # Перевірка кількості параметрів
    print(f"\nКількість параметрів:")
    print(f"Simple Autoencoder: {sum(p.numel() for p in ae.parameters()):,}")
    print(f"VAE: {sum(p.numel() for p in vae.parameters()):,}")
    print(f"Fast Autoencoder: {sum(p.numel() for p in fast_ae.parameters()):,}")