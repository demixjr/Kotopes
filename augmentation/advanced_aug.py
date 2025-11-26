# augmentation/advanced_aug.py
from torchvision import transforms
import torch

class AdvancedAugmentation:
    """Розширена аугментація з використанням PyTorch"""
    
    @staticmethod
    def get_baseline_augmentation(image_size=224):
        """Базові трансформації PyTorch"""
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    @staticmethod
    def get_advanced_augmentation(image_size=224):
        """Розширені трансформації PyTorch"""
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=30),
            transforms.RandomAffine(
                degrees=0, 
                translate=(0.1, 0.1), 
                scale=(0.8, 1.2),
                shear=10
            ),
            transforms.ColorJitter(
                brightness=0.3, 
                contrast=0.3, 
                saturation=0.3, 
                hue=0.2
            ),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    @staticmethod
    def get_light_augmentation(image_size=224):
        """Легка аугментація для валідації"""
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def get_advanced_transforms(image_size=224):
    """Функція для отримання розширених трансформацій"""
    return {
        'train': AdvancedAugmentation.get_advanced_augmentation(image_size),
        'val': AdvancedAugmentation.get_light_augmentation(image_size)
    }

def get_baseline_transforms(image_size=224):
    """Функція для отримання базових трансформацій"""
    return {
        'train': AdvancedAugmentation.get_baseline_augmentation(image_size),
        'val': AdvancedAugmentation.get_light_augmentation(image_size)
    }

if __name__ == "__main__":
    print("Advanced augmentation (PyTorch only):")
    transforms_dict = get_advanced_transforms()
    print(f"Train transforms: {transforms_dict['train']}")
    print(f"Val transforms: {transforms_dict['val']}")