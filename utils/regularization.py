import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

class RegularizationTechniques:
    """Клас для різних методів регуляризації"""
    
    @staticmethod
    def add_dropout(model, dropout_rate=0.5):
        """Додати Dropout шари до моделі"""
        for name, module in model.named_children():
            if isinstance(module, nn.Sequential):
                for child_name, child_module in module.named_children():
                    if isinstance(child_module, nn.Linear):
                        # Додаємо Dropout після Linear шарів
                        new_layers = [child_module, nn.Dropout(dropout_rate)]
                        module[int(child_name)] = nn.Sequential(*new_layers)
            elif isinstance(module, nn.Linear):
                # Додаємо Dropout після основних Linear шарів
                new_layers = [module, nn.Dropout(dropout_rate)]
                setattr(model, name, nn.Sequential(*new_layers))
        return model
    
    @staticmethod
    def get_optimizer_with_l2(model, optimizer_type='adam', weight_decay=1e-4):
        """Отримати оптимізатор з L2 регуляризацією"""
        if optimizer_type.lower() == 'adam':
            return optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)
        elif optimizer_type.lower() == 'sgd':
            return optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=weight_decay)
        else:
            raise ValueError("Підтримуються тільки 'adam' та 'sgd'")
    
    @staticmethod
    def label_smoothing_loss(predictions, targets, smoothing=0.1, num_classes=3):
        """Label Smoothing Loss"""
        confidence = 1.0 - smoothing
        smoothing_value = smoothing / (num_classes - 1)
        
        one_hot = torch.full_like(predictions, smoothing_value)
        one_hot.scatter_(1, targets.unsqueeze(1), confidence)
        
        log_probs = torch.nn.functional.log_softmax(predictions, dim=1)
        loss = torch.mean(torch.sum(-one_hot * log_probs, dim=1))
        return loss
    
    @staticmethod
    def get_scheduler(optimizer, scheduler_type='step', **kwargs):
        """Отримати scheduler для навчання"""
        if scheduler_type == 'step':
            return StepLR(optimizer, step_size=kwargs.get('step_size', 10), 
                         gamma=kwargs.get('gamma', 0.1))
        elif scheduler_type == 'cosine':
            return CosineAnnealingLR(optimizer, T_max=kwargs.get('T_max', 10))
        else:
            return None

class CustomCNNWithRegularization(nn.Module):
    """Custom CNN з розширеними методами регуляризації"""
    
    def __init__(self, num_classes=3, input_size=128, dropout_rate=0.5, use_batchnorm=True):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) if use_batchnorm else nn.Identity()
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64) if use_batchnorm else nn.Identity()
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128) if use_batchnorm else nn.Identity()
        self.pool3 = nn.MaxPool2d(2)
        
        final_size = input_size // 8
        self.fc_input_features = 128 * final_size * final_size
        
        self.fc1 = nn.Linear(self.fc_input_features, 512)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # Conv блок 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # Conv блок 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Conv блок 3
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # FC шари
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x