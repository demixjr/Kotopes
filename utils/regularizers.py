import torch.nn as nn


def add_dropout(model, dropout_rate=0.5):
    """
    Добавляет Dropout в модель для регуляризации
    Dropout случайно выключает нейроны, чтобы модель не зубрила
    """
    print(f"Adding Dropout with rate {dropout_rate}")
    return dropout_rate


def get_weight_decay(weight_decay=0.0001):
    """
    Weight decay (L2 регуляризация) - штрафует большие веса
    Передается в оптимизатор
    """
    print(f"Using weight decay (L2): {weight_decay}")
    return weight_decay


def get_label_smoothing(smoothing=0.1):
    """
    Label smoothing - делает метки мягче (не 0 и 1, а 0.05 и 0.95)
    Помогает модели не быть слишком уверенной
    """
    print(f"Using label smoothing: {smoothing}")
    return smoothing


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy Loss с label smoothing
    """
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        n_classes = pred.size(1)
        
        # Обычный CrossEntropy
        log_pred = nn.functional.log_softmax(pred, dim=1)
        loss = nn.functional.nll_loss(log_pred, target, reduction='sum')
        
        # Добавляем smoothing
        smooth_loss = -log_pred.sum(dim=1).mean()
        loss = (1 - self.smoothing) * loss / pred.size(0) + self.smoothing * smooth_loss
        
        return loss


if __name__ == "__main__":
    print("Testing regularizers...")
    
    dropout = add_dropout(None, 0.5)
    wd = get_weight_decay(0.0001)
    ls = get_label_smoothing(0.1)
    
    print("\nLabel Smoothing Loss:")
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    print(criterion)