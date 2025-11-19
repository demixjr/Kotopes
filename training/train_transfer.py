import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
import copy
import os
import sys

# Добавляем путь к проекту
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transfer_models import get_model
from utils.optimizers import get_optimizer_with_different_lr


def train_model(model, dataloaders, criterion, optimizer, num_epochs=20, 
                device='cuda', patience=5, save_path='checkpoints/transfer_best.pth'):
    """
    Обучение transfer learning модели
    
    Args:
        model: модель для обучения
        dataloaders: dict с 'train' и 'val' DataLoader'ами
        criterion: функция потерь
        optimizer: оптимизатор
        num_epochs: количество эпох
        device: устройство для обучения
        patience: для early stopping
        save_path: путь для сохранения лучшей модели
    """
    
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # История для графиков
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Early stopping
    patience_counter = 0
    best_val_loss = float('inf')
    
    model = model.to(device)
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 50)
        
        # Каждая эпоха имеет фазу обучения и валидации
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            
            # Итерация по данным
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Обнуление градиентов
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward pass + optimize только в train фазе
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Статистика
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Сохраняем историю
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
            
            # Сохраняем лучшую модель
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), save_path)
                print(f'✓ Saved best model with accuracy: {best_acc:.4f}')
            
            # Early stopping
            if phase == 'val':
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    print(f'\nEarly stopping triggered after {epoch+1} epochs')
                    time_elapsed = time.time() - since
                    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
                    print(f'Best val Acc: {best_acc:.4f}')
                    
                    # Загружаем лучшие веса
                    model.load_state_dict(best_model_wts)
                    return model, history
    
    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')
    
    # Загружаем лучшие веса
    model.load_state_dict(best_model_wts)
    return model, history


def plot_training_history(history, save_path='results/transfer/learning_curves.png'):
    """Построение графиков обучения"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f'Saved learning curves to {save_path}')
    plt.close()


def main():
    # Параметры
    DATA_DIR = 'data'
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001
    NUM_CLASSES = 3
    IMAGE_SIZE = 224  # Для pretrained моделей обычно 224
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    
    # Трансформации данных
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    
    # Загрузка данных
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
        for x in ['train', 'val']
    }
    
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, 
                     shuffle=(x == 'train'), num_workers=4)
        for x in ['train', 'val']
    }
    
    print(f"\nDataset sizes:")
    print(f"Train: {len(image_datasets['train'])}")
    print(f"Val: {len(image_datasets['val'])}")
    
    # Выбор модели и режима
    model_name = 'resnet18'  # Можно изменить на 'vgg16' или 'mobilenet_v2'
    mode = 'feature_extraction'  # или 'fine_tuning'
    
    print(f"\nTraining {model_name} in {mode} mode")
    
    # Создание модели
    model = get_model(model_name, num_classes=NUM_CLASSES, pretrained=True, mode=mode)
    
    # Loss и optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.get_trainable_params(), lr=LEARNING_RATE)
    
    # Обучение
    save_path = f'checkpoints/transfer_{model_name}_{mode}_best.pth'
    model, history = train_model(
        model, dataloaders, criterion, optimizer,
        num_epochs=NUM_EPOCHS, device=device, 
        patience=5, save_path=save_path
    )
    
    # Сохранение графиков
    plot_path = f'results/transfer/{model_name}_{mode}_curves.png'
    plot_training_history(history, save_path=plot_path)
    
    print("\nTraining completed!")


if __name__ == '__main__':
    main()