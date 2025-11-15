import torch
import torch.nn as nn

class CustomCNN(nn.Module):
    def __init__(self, num_classes=3, input_size=128):
        super(CustomCNN, self).__init__()
        
        # Розрахунок розміру після згорткових шарів
        self.feature_size = self._calculate_feature_size(input_size)
        
        # Перший блок: Conv -> BatchNorm -> ReLU -> MaxPool
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Другий блок
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Третій блок
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Четвертий блок
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Повністю з'єднані шари
        self.fc1 = nn.Linear(256 * self.feature_size * self.feature_size, 512)
        self.relu5 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
    def _calculate_feature_size(self, input_size):
        """Розраховує розмір feature map після всіх пулінгів"""
        size = input_size
        for _ in range(4):  # 4 пулінгових шари
            size = size // 2
        return size
        
    def forward(self, x):
        # Прямий прохід через згорткові шари
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        
        # Вирівнювання для повнозв'язних шарів
        x = x.view(x.size(0), -1)
        
        # Повнозв'язні шари
        x = self.relu5(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def initialize_model(num_classes=3, input_size=128):
    return CustomCNN(num_classes=num_classes, input_size=input_size)

if __name__ == "__main__":
    # Тестування моделі
    model = CustomCNN(num_classes=3, input_size=128)
    sample_input = torch.randn(1, 3, 128, 128)
    output = model(sample_input)
    print(f"Розмір виходу: {output.shape}")
    print(f"Кількість параметрів: {sum(p.numel() for p in model.parameters()):,}")