import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
import matplotlib.pyplot as plt
import time
import copy
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transfer_models import get_model
from utils.regularizers import LabelSmoothingCrossEntropy
from augmentation.baseline_aug import get_baseline_transforms
from augmentation.advanced_aug import get_advanced_transforms


def train_with_regularization(model, dataloaders, criterion, optimizer, 
                              num_epochs=5, device='cuda', patience=3,
                              save_path='checkpoints/augmented_best.pth'):
    """
    Простое обучение с регуляризацией
    """
    
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    patience_counter = 0
    best_val_loss = float('inf')
    
    model = model.to(device)
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 50)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), save_path)
                print(f'✓ Saved best model: {best_acc:.4f}')
            
            if phase == 'val':
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    print(f'\nEarly stopping at epoch {epoch+1}')
                    model.load_state_dict(best_model_wts)
                    return model, history
    
    time_elapsed = time.time() - since
    print(f'\nTraining done in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')
    
    model.load_state_dict(best_model_wts)
    return model, history


def main():
    # Параметры
    DATA_DIR = 'data'
    BATCH_SIZE = 32
    NUM_EPOCHS = 5
    NUM_CLASSES = 3
    IMAGE_SIZE = 224
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')
    
    # Выбираем аугментацию
    print("\n=== Training with BASELINE augmentation ===")
    transforms_dict = get_baseline_transforms(IMAGE_SIZE)
    
    # Загружаем данные
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(DATA_DIR, x), transforms_dict[x])
        for x in ['train', 'val']
    }
    
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, 
                     shuffle=(x == 'train'), num_workers=0)
        for x in ['train', 'val']
    }
    
    print(f"Train: {len(image_datasets['train'])}")
    print(f"Val: {len(image_datasets['val'])}")
    
    # Создаем модель
    model = get_model('resnet18', num_classes=NUM_CLASSES, 
                     pretrained=True, mode='fine_tuning')
    
    # Регуляризация
    # 1. Dropout уже есть в ResNet
    # 2. Weight decay в оптимизаторе
    # 3. Label smoothing в loss
    
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
    
    # Обучение
    model, history = train_with_regularization(
        model, dataloaders, criterion, optimizer,
        num_epochs=NUM_EPOCHS, device=device,
        save_path='checkpoints/augmented_best.pth'
    )
    
    # Сохраняем график
    os.makedirs('results/augmentation', exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'], label='Val')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history['train_acc'], label='Train')
    ax2.plot(history['val_acc'], label='Val')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/augmentation/training_curves.png')
    print("\nSaved to results/augmentation/training_curves.png")


if __name__ == '__main__':
    main()