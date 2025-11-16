import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomCNN(nn.Module):
    def __init__(self, num_classes=3, input_size=128):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # 3x3 kernel
        self.pool1 = nn.MaxPool2d(2)  # зменшення в 2 рази
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        
        # Один повнозв'язний шар
        self.fc = nn.Linear(64 * 32 * 32, num_classes)  # для 128x128 зображень
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Conv block 1
        x = self.pool1(F.relu(self.conv1(x)))  # 128x128 -> 64x64
        
        # Conv block 2  
        x = self.pool2(F.relu(self.conv2(x)))  # 64x64 -> 32x32
        
        # Classifier
        x = x.view(x.size(0), -1)  # flatten
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

def initialize_model(num_classes=3, input_size=64):
    return CustomCNN(num_classes=num_classes, input_size=input_size)

if __name__ == "__main__":
    # Тестування моделі
    model = CustomCNN(num_classes=3, input_size=64)
    sample_input = torch.randn(1, 3, 64, 64)
    output = model(sample_input)
    print(f"Розмір виходу: {output.shape}")
    print(f"Кількість параметрів: {sum(p.numel() for p in model.parameters()):,}")