from torchvision import transforms

def get_baseline_transforms(image_size=224):
    """
    Базовая аугментация - простые трансформации
    """
    
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),  # Отражение по горизонтали
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
    print("Baseline augmentation:")
    transforms_dict = get_baseline_transforms()
    print(f"Train transforms: {transforms_dict['train']}")
    print(f"Val transforms: {transforms_dict['val']}")