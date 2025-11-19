import torch.optim as optim


def get_optimizer_with_different_lr(model, base_lr=0.001, classifier_lr=0.01, 
                                    optimizer_type='adam'):
    """
    Создает оптимизатор с разными learning rates для разных частей модели
    
    Args:
        model: модель (должна иметь атрибуты features и classifier/fc)
        base_lr: learning rate для базовых слоев (features)
        classifier_lr: learning rate для классификатора (обычно выше)
        optimizer_type: 'adam' или 'sgd'
    
    Returns:
        optimizer: настроенный оптимизатор
    """
    
    # Разделяем параметры на группы
    feature_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # Определяем, к какой группе относится параметр
        if 'fc' in name or 'classifier' in name:
            classifier_params.append(param)
        else:
            feature_params.append(param)
    
    # Создаем группы параметров с разными lr
    param_groups = []
    
    if len(feature_params) > 0:
        param_groups.append({
            'params': feature_params,
            'lr': base_lr
        })
    
    if len(classifier_params) > 0:
        param_groups.append({
            'params': classifier_params,
            'lr': classifier_lr
        })
    
    # Создаем оптимизатор
    if optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(param_groups)
    elif optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(param_groups, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    print(f"\nOptimizer created with different learning rates:")
    print(f"  Features LR: {base_lr}")
    print(f"  Classifier LR: {classifier_lr}")
    print(f"  Total trainable params: {len(feature_params) + len(classifier_params)}")
    
    return optimizer


def get_optimizer_simple(model, lr=0.001, optimizer_type='adam'):
    """
    Создает простой оптимизатор с одним learning rate
    
    Args:
        model: модель
        lr: learning rate
        optimizer_type: 'adam' или 'sgd'
    
    Returns:
        optimizer: настроенный оптимизатор
    """
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    if optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(trainable_params, lr=lr)
    elif optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(trainable_params, lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    print(f"\nOptimizer created:")
    print(f"  Type: {optimizer_type}")
    print(f"  Learning rate: {lr}")
    print(f"  Total trainable params: {len(trainable_params)}")
    
    return optimizer


def get_lr_scheduler(optimizer, scheduler_type='step', **kwargs):
    """
    Создает learning rate scheduler
    
    Args:
        optimizer: оптимизатор
        scheduler_type: 'step', 'plateau', 'cosine'
        **kwargs: дополнительные параметры для scheduler
    
    Returns:
        scheduler: настроенный scheduler
    """
    
    if scheduler_type == 'step':
        # Уменьшает LR каждые step_size эпох
        step_size = kwargs.get('step_size', 7)
        gamma = kwargs.get('gamma', 0.1)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        
    elif scheduler_type == 'plateau':
        # Уменьшает LR когда метрика перестает улучшаться
        mode = kwargs.get('mode', 'min')
        factor = kwargs.get('factor', 0.1)
        patience = kwargs.get('patience', 3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience
        )
        
    elif scheduler_type == 'cosine':
        # Косинусный scheduler
        T_max = kwargs.get('T_max', 10)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
        
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    print(f"Scheduler created: {scheduler_type}")
    
    return scheduler


if __name__ == "__main__":
    # Тестирование
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from models.transfer_models import get_model
    
    print("Testing optimizer with different learning rates...")
    model = get_model('resnet18', num_classes=3, mode='fine_tuning')
    
    # Тест с разными LR
    optimizer = get_optimizer_with_different_lr(
        model, 
        base_lr=0.0001, 
        classifier_lr=0.001,
        optimizer_type='adam'
    )
    
    print("\nTesting simple optimizer...")
    optimizer_simple = get_optimizer_simple(model, lr=0.001, optimizer_type='sgd')
    
    print("\nTesting scheduler...")
    scheduler = get_lr_scheduler(optimizer, scheduler_type='step', step_size=5)