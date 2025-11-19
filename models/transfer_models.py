import torch
import torch.nn as nn
from torchvision import models


class TransferModel(nn.Module):
    """Базовый класс для transfer learning моделей"""
    
    def __init__(self, num_classes=3, pretrained=True):
        super(TransferModel, self).__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        
    def freeze_features(self):
        """Заморозка всех слоев кроме классификатора (для feature extraction)"""
        for param in self.features.parameters():
            param.requires_grad = False
            
    def unfreeze_features(self):
        """Разморозка всех слоев (для fine-tuning)"""
        for param in self.features.parameters():
            param.requires_grad = True
            
    def get_trainable_params(self):
        """Получить обучаемые параметры"""
        return [p for p in self.parameters() if p.requires_grad]


class ResNet18Transfer(TransferModel):
    """ResNet18 для transfer learning"""
    
    def __init__(self, num_classes=3, pretrained=True):
        super(ResNet18Transfer, self).__init__(num_classes, pretrained)
        
        # Загружаем pretrained ResNet18
        self.model = models.resnet18(pretrained=pretrained)
        
        # Сохраняем features для возможности заморозки
        self.features = nn.Sequential(*list(self.model.children())[:-1])
        
        # Заменяем последний fully connected слой
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.model(x)
    
    def freeze_features(self):
        """Заморозка всех слоев кроме fc"""
        for name, param in self.model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
                
    def unfreeze_features(self):
        """Разморозка всех слоев"""
        for param in self.model.parameters():
            param.requires_grad = True


class VGG16Transfer(TransferModel):
    """VGG16 для transfer learning"""
    
    def __init__(self, num_classes=3, pretrained=True):
        super(VGG16Transfer, self).__init__(num_classes, pretrained)
        
        # Загружаем pretrained VGG16
        self.model = models.vgg16(pretrained=pretrained)
        
        # Сохраняем features
        self.features = self.model.features
        
        # Заменяем последний classifier слой
        num_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.model(x)
    
    def freeze_features(self):
        """Заморозка features, оставляем только classifier"""
        for param in self.model.features.parameters():
            param.requires_grad = False
            
    def unfreeze_features(self):
        """Разморозка всех слоев"""
        for param in self.model.parameters():
            param.requires_grad = True


class MobileNetV2Transfer(TransferModel):
    """MobileNetV2 для transfer learning (бонусная легкая модель)"""
    
    def __init__(self, num_classes=3, pretrained=True):
        super(MobileNetV2Transfer, self).__init__(num_classes, pretrained)
        
        # Загружаем pretrained MobileNetV2
        self.model = models.mobilenet_v2(pretrained=pretrained)
        
        # Сохраняем features
        self.features = self.model.features
        
        # Заменяем последний classifier слой
        num_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.model(x)
    
    def freeze_features(self):
        """Заморозка features"""
        for param in self.model.features.parameters():
            param.requires_grad = False
            
    def unfreeze_features(self):
        """Разморозка всех слоев"""
        for param in self.model.parameters():
            param.requires_grad = True


def get_model(model_name, num_classes=3, pretrained=True, mode='feature_extraction'):
    """
    Получить модель по имени
    
    Args:
        model_name: 'resnet18', 'vgg16', или 'mobilenet_v2'
        num_classes: количество классов
        pretrained: использовать pretrained веса
        mode: 'feature_extraction' или 'fine_tuning'
    """
    
    if model_name == 'resnet18':
        model = ResNet18Transfer(num_classes, pretrained)
    elif model_name == 'vgg16':
        model = VGG16Transfer(num_classes, pretrained)
    elif model_name == 'mobilenet_v2':
        model = MobileNetV2Transfer(num_classes, pretrained)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Устанавливаем режим
    if mode == 'feature_extraction':
        model.freeze_features()
    elif mode == 'fine_tuning':
        model.unfreeze_features()
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'feature_extraction' or 'fine_tuning'")
    
    return model


if __name__ == "__main__":
    # Тестирование моделей
    print("Testing ResNet18...")
    model = get_model('resnet18', num_classes=3, mode='feature_extraction')
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"Output shape: {out.shape}")
    
    print("\nTesting VGG16...")
    model = get_model('vgg16', num_classes=3, mode='fine_tuning')
    out = model(x)
    print(f"Output shape: {out.shape}")
    
    print("\nTesting MobileNetV2...")
    model = get_model('mobilenet_v2', num_classes=3, mode='feature_extraction')
    out = model(x)
    print(f"Output shape: {out.shape}")