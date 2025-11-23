from torchvision import transforms

def get_advanced_transforms(image_size=224):
    """
    Продвинутая аугментация - больше трансформаций для обобщения
    """
    
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),  # Отражение
        transforms.RandomRotation(15),  # Поворот до 15 градусов
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Изменение цветов
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Небольшой сдвиг
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return {
        'train': train_transform,
        'val': val_transform
    }


if __name__ == "__main__":
    print("Advanced augmentation:")
    transforms_dict = get_advanced_transforms()
    print(f"Train transforms: {transforms_dict['train']}")
    print(f"Val transforms: {transforms_dict['val']}")