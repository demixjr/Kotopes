import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

def get_data_loaders(data_dir, input_size=128, batch_size=32):
    """
    Створює DataLoader'и для навчання та валідації.
    """
    # Трансформації для навчальних даних
    train_transforms = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Трансформації для валідаційних даних
    val_transforms = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Створення datasets
    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'train'),
        transform=train_transforms
    )
    
    val_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'val'),
        transform=val_transforms
    )

    # Створення DataLoader'ів
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    # Отримання назв класів
    class_names = train_dataset.classes

    print(f"Класи: {class_names}")
    print(f"Навчальні зображення: {len(train_dataset)}")
    print(f"Валідаційні зображення: {len(val_dataset)}")

    return train_loader, val_loader, class_names



if __name__ == "__main__":
    data_directory = "D:\python\Kotopes\data"  
    train_loader, val_loader, classes = get_data_loaders(data_directory)