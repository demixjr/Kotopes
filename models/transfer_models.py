import torch
import torch.nn as nn
from torchvision import models
class TransferModel(nn.Module):
    """Базовий клас для transfer learning моделей"""
    def __init__(self, num_classes=3, pretrained=True):
        super(TransferModel, self).__init__()  # Виклик конструктора батьківського класу
        self.num_classes = num_classes         # Кількість класів (3 для кіт/собака/птиця)
        self.pretrained = pretrained           # Чи використовувати попередньо навчені ваги
        
    def freeze_features(self):
        """Заморозити всі шари крім класифікатора"""
        raise NotImplementedError
        
    def unfreeze_features(self):
        """Розморозити всі шари"""
        for param in self.parameters():
            param.requires_grad = True
            
    def get_trainable_params(self):
        """Отримати параметри для навчання"""
        return [p for p in self.parameters() if p.requires_grad]


class ResNet18Transfer(TransferModel):
    """ResNet18 для transfer learning"""
    
    def __init__(self, num_classes=3, pretrained=True):
        super(ResNet18Transfer, self).__init__(num_classes, pretrained)
        self.model = models.resnet18(pretrained=pretrained)  # Завантажуємо ResNet18 з PyTorch
        num_features = self.model.fc.in_features             # Дізнаємось розмір останнього шару
        self.model.fc = nn.Linear(num_features, num_classes) # Замінюємо останній шар під наші класи
        
    def forward(self, x):
        return self.model(x)
    
    def freeze_features(self):
     for name, param in self.model.named_parameters():  # Ітеруємо по всіх параметрах
        if 'fc' not in name:                           # Якщо не останній шар (fc = fully connected)
            param.requires_grad = False                # Заморожуємо - не буде оновлюватись при навчанні

class ResNet50Transfer(TransferModel):
    """ResNet50 для transfer learning"""
    
    def __init__(self, num_classes=3, pretrained=True):
        super(ResNet50Transfer, self).__init__(num_classes, pretrained)
        
        self.model = models.resnet50(pretrained=pretrained)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.model(x)
    
    def freeze_features(self):
        for name, param in self.model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False


class EfficientNetB0Transfer(TransferModel):
    """EfficientNet-B0 для transfer learning"""
    
    def __init__(self, num_classes=3, pretrained=True):
        super(EfficientNetB0Transfer, self).__init__(num_classes, pretrained)
        
        self.model = models.efficientnet_b0(pretrained=pretrained)
        num_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.model(x)
    
    def freeze_features(self):
        for name, param in self.model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False


class MobileNetV2Transfer(TransferModel):
    """MobileNetV2 для transfer learning"""
    
    def __init__(self, num_classes=3, pretrained=True):
        super(MobileNetV2Transfer, self).__init__(num_classes, pretrained)
        
        self.model = models.mobilenet_v2(pretrained=pretrained)
        num_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.model(x)
    
    def freeze_features(self):
        for name, param in self.model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False


def get_model(model_name, num_classes=3, pretrained=True, mode='feature_extraction'):
    """
    Фабрика моделей для transfer learning
    
    Args:
        model_name: 'resnet18', 'resnet50', 'efficientnet_b0', 'mobilenet_v2'
        num_classes: кількість класів
        pretrained: використовувати pretrained ваги
        mode: 'feature_extraction' або 'fine_tuning'
    """
    
    model_dict = {
        'resnet18': ResNet18Transfer,
        'resnet50': ResNet50Transfer,
        'efficientnet_b0': EfficientNetB0Transfer,
        'mobilenet_v2': MobileNetV2Transfer
    }
    
    if model_name not in model_dict:
        raise ValueError(f"Невідома модель: {model_name}")
    
    model = model_dict[model_name](num_classes, pretrained)
    
    # Встановлення режиму
    if mode == 'feature_extraction':
        model.freeze_features()
    elif mode == 'fine_tuning':
        model.unfreeze_features()
    else:
        raise ValueError("Режим повинен бути 'feature_extraction' або 'fine_tuning'")
    
    return model


if __name__ == "__main__":
    # Тестування моделей
    print("Тестування моделей transfer learning...")
    
    test_models = ['resnet18', 'resnet50', 'efficientnet_b0', 'mobilenet_v2']
    
    for model_name in test_models:
        print(f"\n {model_name}:")
        
        # Feature extraction
        model_fe = get_model(model_name, num_classes=3, mode='feature_extraction')
        trainable_params = sum(p.numel() for p in model_fe.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model_fe.parameters())
        
        print(f"   Feature Extraction: {trainable_params:,} / {total_params:,} параметрів")
        
        # Fine-tuning
        model_ft = get_model(model_name, num_classes=3, mode='fine_tuning')
        trainable_params = sum(p.numel() for p in model_ft.parameters() if p.requires_grad)
        
        print(f"   Fine-tuning: {trainable_params:,} / {total_params:,} параметрів")
    
    print("\nВсі моделі успішно створені!")