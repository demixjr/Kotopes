import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomCNN(nn.Module):
    def __init__(self, num_classes=3, input_size=128):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # 3 канали (RGB) на вході, 32 фільтри на виході, 3x3 розмір фільтра
        self.pool1 = nn.MaxPool2d(2)  # зменшення зображення у 2 разии 128x128 -> 64x64
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # 32 канали на вході, 64 фільтри на виході, 3x3 розмір фільтра
        self.pool2 = nn.MaxPool2d(2)  # зменшення зображення у 2 рази 64x64 -> 32x32
        
        # Автоматичний розрахунок розміру для final classifier шару
        final_size = input_size // 4  # 128 / 2 / 2 = 32
        self.fc_input_features = 64 * final_size * final_size  # 64 * 32 * 32 = 65536
        
        self.fc1 = nn.Linear(self.fc_input_features, 512)  # 65K → 512
        self.fc2 = nn.Linear(512, num_classes)             # 512 → 3
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # згортковий блок 1
        x = self.pool1(F.relu(self.conv1(x)))  # 128x128 -> 64x64
        
        # згортковий блок 2  
        x = self.pool2(F.relu(self.conv2(x)))  # 64x64 -> 32x32
        
        # класифікаційний блок
        x = x.view(x.size(0), -1)  # flatten
        x = self.dropout(F.relu(self.fc1(x)))  # FC1 + ReLU + Dropout
        x = self.fc2(x)
        
        return x

def initialize_model(num_classes=3, input_size=128):  # Змінено на 128
    return CustomCNN(num_classes=num_classes, input_size=input_size)

if __name__ == "__main__":
    # Тестування моделі з 128x128
    model = CustomCNN(num_classes=3, input_size=128)
    sample_input = torch.randn(1, 3, 128, 128)  
    output = model(sample_input)
    print(f"Розмір виходу: {output.shape}")  # Має бути [1, 3]
    print(f"Кількість параметрів: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Розмір після flatten: {model.fc_input_features}")  # Має бути 65536