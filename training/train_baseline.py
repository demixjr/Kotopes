import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  
sys.path.insert(0, project_root)

from models.custom_cnn import CustomCNN
from utils.data_loader import get_data_loaders
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  

DATA_DIR = os.path.join(project_root, "data")
CHECKPOINTS_DIR = os.path.join(project_root, "checkpoints") 
RESULTS_DIR = os.path.join(project_root, "results")

class BaselineTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        
        # Історія тренування
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_accuracy = 0
        
    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (data, targets) in enumerate(pbar):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Forward pass
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Статистика
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Оновлення прогресу
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_accuracy = 100. * correct / total
        
        return epoch_loss, epoch_accuracy
    
    def validate_epoch(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in tqdm(self.val_loader, desc="Validation"):
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_accuracy = 100. * correct / total
        
        return epoch_loss, epoch_accuracy
    
    def train(self, epochs, early_stopping_patience=5):
        print(f"Початок навчання на {epochs} епох.")
        print(f"Рання зупинка після {early_stopping_patience} епох без покращення")
        
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\n{'='*60}")
            print(f"Епоха {epoch+1}/{epochs}")
            print(f"{'='*60}")
            
            # Навчання
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Валідація
            val_loss, val_acc = self.validate_epoch()
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            print(f"\n Результати епохи {epoch+1}:")
            print(f"   Навчання - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
            print(f"   Валідація - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
            
            # Перевірка найкращої моделі
            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                patience_counter = 0
                self.save_checkpoint(epoch)
                print(f"Нова найкраща модель. Accuracy: {val_acc:.2f}%")
            else:
                patience_counter += 1
                print(f"Без покращення: {patience_counter}/{early_stopping_patience}")
            
            # Рання зупинка
            if patience_counter >= early_stopping_patience:
                print(f"Рання зупинка на епосі {epoch+1}")
                break
        
        # Завантаження найкращої моделі
        self.load_best_checkpoint()
        return self.best_val_accuracy
    
    def save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_accuracy': self.best_val_accuracy,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
        }
        
        os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
        torch.save(checkpoint, os.path.join(CHECKPOINTS_DIR, 'baseline_best.pth'))
    
    def load_best_checkpoint(self):
        checkpoint_path = os.path.join(CHECKPOINTS_DIR, 'baseline_best.pth')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("Найкраща модель завантажена!")
    
    def plot_learning_curves(self):
        os.makedirs(os.path.join(RESULTS_DIR, 'baseline'), exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Графік втрат
        ax1.plot(self.train_losses, label='Train Loss', linewidth=2)
        ax1.plot(self.val_losses, label='Val Loss', linewidth=2)
        ax1.set_title('Learning Curves - Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Графік точності
        ax2.plot(self.train_accuracies, label='Train Accuracy', linewidth=2)
        ax2.plot(self.val_accuracies, label='Val Accuracy', linewidth=2)
        ax2.set_title('Learning Curves - Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Цільова лінія 60%
        ax2.axhline(y=60, color='red', linestyle='--', alpha=0.7, label='Target 60%')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'baseline', 'learning_curves.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self):
        """Зберігає результати у текстовий файл"""
        results_dir = os.path.join(RESULTS_DIR, 'baseline')
        os.makedirs(results_dir, exist_ok=True)
        
        with open(os.path.join(results_dir, 'training_results.txt'), 'w') as f:
            f.write("=== РЕЗУЛЬТАТИ НАВЧАННЯ BASELINE CNN ===\n\n")
            f.write(f"Найкраща точність на валідації: {self.best_val_accuracy:.2f}%\n")
            f.write(f"Фінальна точність на навчанні: {self.train_accuracies[-1]:.2f}%\n")
            f.write(f"Фінальна точність на валідації: {self.val_accuracies[-1]:.2f}%\n")
            f.write(f"Кількість епох: {len(self.train_accuracies)}\n\n")
            
            f.write("Історія навчання:\n")
            f.write("Epoch | Train Loss | Val Loss | Train Acc | Val Acc\n")
            f.write("-" * 50 + "\n")
            for i in range(len(self.train_losses)):
                f.write(f"{i+1:3d}   | {self.train_losses[i]:.4f}    | {self.val_losses[i]:.4f}  | "
                       f"{self.train_accuracies[i]:6.2f}%  | {self.val_accuracies[i]:6.2f}%\n")

def main():
    # Параметри
    CONFIG = {
        'batch_size': 32,
        'epochs': 20,
        'learning_rate': 0.001,
        'input_size': 128,
        'early_stopping_patience': 5
    }
    
    # Пристрій
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Використовується пристрій: {device}")
    
    # Завантаження даних
    print("📁 Завантаження даних...")
    train_loader, val_loader, class_names = get_data_loaders(
        DATA_DIR, 
        input_size=CONFIG['input_size'], 
        batch_size=CONFIG['batch_size']
    )
    
    # Модель
    print("🔄 Ініціалізація моделі...")
    model = CustomCNN(
        num_classes=len(class_names), 
        input_size=CONFIG['input_size']
    ).to(device)
    
    # Оптимізатор та функція втрат
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    # Тренер
    trainer = BaselineTrainer(model, train_loader, val_loader, criterion, optimizer, device)
    
    # Навчання
    best_accuracy = trainer.train(
        epochs=CONFIG['epochs'],
        early_stopping_patience=CONFIG['early_stopping_patience']
    )
    
    # Візуалізація та збереження результатів
    trainer.plot_learning_curves()
    trainer.save_results()
    
    # Фінальний результат
    print(f"\n{'='*50}")
    print("🎯 ФІНАЛЬНІ РЕЗУЛЬТАТИ")
    print(f"{'='*50}")
    print(f"Найкраща точність на валідації: {best_accuracy:.2f}%")
    
    if best_accuracy >= 60:
        print("ЦІЛЬ ДОСЯГНУТА! Accuracy ≥ 60%")
    else:
        print(" Ціль не досягнута. Потрібно покращити модель.")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()