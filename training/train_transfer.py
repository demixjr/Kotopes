import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transfer_models import get_model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
import json
from datetime import datetime
from tqdm import tqdm

class TransferLearningTrainer:
    def __init__(self, data_dir='data', batch_size=32, image_size=224, num_classes=3):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'learning_rates': []
        }
        self.best_val_accuracy = 0.0
        self.best_model_path = ""
        
        self._create_directories()
        
        print("=" * 70)
        print("TRANSFER LEARNING TRAINER")
        print("=" * 70)

    def _create_directories(self):
        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('results/transfer', exist_ok=True)

    def _print_header(self, text):
        print(f"\n{'='*60}")
        print(f"{text}")
        print(f"{'='*60}")

    def _print_epoch_info(self, epoch, total_epochs, train_loss, train_acc, val_loss, val_acc, lr, epoch_time, remaining_time):
        print(f"Епоха {epoch+1:02d}/{total_epochs:02d} | "
              f"Train: {train_loss:.4f} loss, {train_acc:.2f}% acc | "
              f"Val: {val_loss:.4f} loss, {val_acc:.2f}% acc | "
              f"LR: {lr:.2e} | "
              f"Час: {epoch_time:.1f}с | "
              f"Залишилось: ~{remaining_time:.1f}с")

    def setup_data(self):
        self._print_header("ЗАВАНТАЖЕННЯ ТА ПІДГОТОВКА ДАНИХ")
        
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ]),
            'val': transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ])
        }
        
        try:
            self.datasets = {
                x: datasets.ImageFolder(os.path.join(self.data_dir, x), data_transforms[x])
                for x in ['train', 'val']
            }
            
            self.dataloaders = {
                x: DataLoader(self.datasets[x], batch_size=self.batch_size,
                            shuffle=(x == 'train'), num_workers=0)
                for x in ['train', 'val']
            }
            
            self.class_names = self.datasets['train'].classes
            
            print("Дані успішно завантажено!")
            print(f"Навчальний набор: {len(self.datasets['train'])} зображень")
            print(f"Валідаційний набор: {len(self.datasets['val'])} зображень")
            print(f"Класи: {', '.join(self.class_names)}")
            
        except Exception as e:
            print(f"Помилка завантаження даних: {e}")
            raise

    def create_model(self, model_name='resnet18', mode='feature_extraction', learning_rate=0.001):
        self._print_header(f"СТВОРЕННЯ МОДЕЛІ: {model_name.upper()} ({mode})")
        
        try:
            self.model = get_model(model_name, self.num_classes, True, mode)
            self.model = self.model.to(self.device)
            print(f"Модель {model_name} завантажено!")
            
        except Exception as e:
            print(f"Помилка створення моделі: {e}")
            raise

        feature_params = []
        classifier_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'fc' in name or 'classifier' in name:
                    classifier_params.append(param)
                else:
                    feature_params.append(param)
        
        self.optimizer = torch.optim.SGD([
            {'params': feature_params, 'lr': 0.00001},  
            {'params': classifier_params, 'lr': 0.00005}   
        ], momentum=0.8, weight_decay=0.0001)
        
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=2, gamma=0.1)
        
        print(f"Learning rate: {learning_rate}")
        print(f"Параметри для навчання: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

    def train(self, num_epochs=10, patience=3):
        self._print_header("ПОЧАТОК НАВЧАННЯ")
        
        since = time.time()
        best_acc = 0.0
        patience_counter = 0
        epoch_times = []
        
        print(f"Заплановано епох: {num_epochs}")
        print(f"Рання зупинка через: {patience} епох без покращення")
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rates'].append(current_lr)
            
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                    desc = f" Епоха {epoch+1}/{num_epochs} - Тренування"
                else:
                    self.model.eval()
                    desc = f" Епоха {epoch+1}/{num_epochs} - Валідація"
                
                running_loss = 0.0
                running_corrects = 0
                
                pbar = tqdm(self.dataloaders[phase], desc=desc, leave=False)
                
                for inputs, labels in pbar:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    self.optimizer.zero_grad()
                    
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)
                        
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                    
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    
                    current_acc = running_corrects.double() / (pbar.n * inputs.size(0)) * 100
                    pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Acc': f'{current_acc:.1f}%'
                    })
                
                pbar.close()
                
                epoch_loss = running_loss / len(self.datasets[phase])
                epoch_acc = (running_corrects.double() / len(self.datasets[phase])) * 100
                
                if phase == 'train':
                    self.history['train_loss'].append(epoch_loss)
                    self.history['train_acc'].append(epoch_acc.item())
                    train_loss, train_acc = epoch_loss, epoch_acc.item()
                else:
                    self.history['val_loss'].append(epoch_loss)
                    self.history['val_acc'].append(epoch_acc.item())
                    val_loss, val_acc = epoch_loss, epoch_acc.item()
            
            epoch_time = time.time() - epoch_start_time
            epoch_times.append(epoch_time)
            avg_epoch_time = sum(epoch_times) / len(epoch_times)
            remaining_epochs = num_epochs - (epoch + 1)
            remaining_time = remaining_epochs * avg_epoch_time
            
            self._print_epoch_info(epoch, num_epochs, train_loss, train_acc, val_loss, val_acc, 
                                 current_lr, epoch_time, remaining_time)
            
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                self.best_val_accuracy = best_acc
                self.best_model_path = 'checkpoints/transfer_best.pth'
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f" Збережено найкращу модель! Точність: {best_acc:.2f}%")
            else:
                patience_counter += 1
                print(f" Без покращення: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f" Early stopping на епосі {epoch+1}")
                break
            
            if phase == 'train':
                self.scheduler.step()
        
        time_elapsed = time.time() - since
        print(f"\n Навчання завершено за {time_elapsed//60:.0f}хв {time_elapsed%60:.0f}с")
        print(f" Найкраща точність валідації: {best_acc:.2f}%")

    def save_results(self, model_name, mode):
        experiment_folder = f"results/transfer/{model_name}_{mode}"
        os.makedirs(experiment_folder, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{model_name}_{mode}_{timestamp}"
        
        self._print_header("ЗБЕРЕЖЕННЯ РЕЗУЛЬТАТІВ")
        
        txt_filename = f"{base_filename}_results.txt"
        txt_path = os.path.join(experiment_folder, txt_filename)
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("         РЕЗУЛЬТАТИ НАВЧАННЯ TRANSFER LEARNING\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Модель: {model_name.upper()}\n")
            f.write(f"Режим: {mode.upper()}\n")
            f.write(f"Дата навчання: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Розмір зображення: {self.image_size}x{self.image_size}\n")
            f.write(f"Batch size: {self.batch_size}\n")
            f.write(f"Кількість класів: {self.num_classes}\n\n")
            
            f.write("МЕТРИКИ НАВЧАННЯ:\n")
            f.write(f"   Найкраща точність на валідації: {self.best_val_accuracy:.2f}%\n")
            f.write(f"   Фінальна точність на навчанні: {self.history['train_acc'][-1]:.2f}%\n")
            f.write(f"   Фінальна точність на валідації: {self.history['val_acc'][-1]:.2f}%\n")
            f.write(f"   Фінальний loss на навчанні: {self.history['train_loss'][-1]:.4f}\n")
            f.write(f"   Фінальний loss на валідації: {self.history['val_loss'][-1]:.4f}\n")
            f.write(f"   Кількість епох: {len(self.history['train_acc'])}\n\n")
            
            f.write("ІСТОРІЯ НАВЧАННЯ:\n")
            f.write("Epoch | Train Loss | Val Loss | Train Acc | Val Acc | LR\n")
            f.write("-" * 65 + "\n")
            
            for i in range(len(self.history['train_acc'])):
                f.write(f"{i+1:3d}   | {self.history['train_loss'][i]:10.4f} | {self.history['val_loss'][i]:8.4f} | "
                       f"{self.history['train_acc'][i]:9.2f}% | {self.history['val_acc'][i]:7.2f}% | "
                       f"{self.history['learning_rates'][i]:.2e}\n")
        
        json_filename = f"{base_filename}_history.json"
        json_path = os.path.join(experiment_folder, json_filename)
        
        json_results = {
            'model_name': model_name,
            'mode': mode,
            'best_val_accuracy': float(self.best_val_accuracy),
            'training_history': self.history,
            'training_parameters': {
                'image_size': self.image_size,
                'batch_size': self.batch_size,
                'num_classes': self.num_classes,
                'device': str(self.device)
            },
            'timestamp': timestamp
        }
        
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        plot_filename = f"{base_filename}_plots.png"
        plot_path = os.path.join(experiment_folder, plot_filename)
        self._plot_training_history(plot_path)
        
        model_filename = f"{base_filename}_best_model.pth"
        model_path = os.path.join(experiment_folder, model_filename)
        torch.save(self.model.state_dict(), model_path)
        
        print(f"Результати збережено в папці: {experiment_folder}/")
        print(f"Текстові результати: {txt_filename}")
        print(f"JSON історія: {json_filename}")
        print(f"Графіки: {plot_filename}")
        print(f"Найкраща модель: {model_filename}")

    def _plot_training_history(self, save_path):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(epochs, self.history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
        ax2.plot(epochs, self.history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def find_existing_result(model_name, mode):
    import glob
    pattern = f"results/transfer/{model_name}_{mode}/*_results.txt"
    result_files = glob.glob(pattern)
    
    if not result_files:
        return None
    
    latest_file = max(result_files, key=os.path.getctime)
    
    try:
        with open(latest_file, 'r', encoding='utf-8') as f:
            content = f.read()
            for line in content.split('\n'):
                if 'Найкраща точність на валідації:' in line:
                    accuracy = float(line.split(':')[1].strip().replace('%', ''))
                    return {
                        'accuracy': accuracy,
                        'file': latest_file,
                        'timestamp': os.path.getctime(latest_file)
                    }
    except Exception as e:
        print(f"Помилка читання файлу {latest_file}: {e}")
    
    return None

def should_skip_experiment(model_name, mode, min_accuracy=0.0):
    existing_result = find_existing_result(model_name, mode)
    
    if existing_result:
        print(f"   Знайдено існуючий результат для {model_name} {mode}:")
        print(f"   Точність: {existing_result['accuracy']:.2f}%")
        print(f"   Файл: {os.path.basename(existing_result['file'])}")
        
        if existing_result['accuracy'] >= min_accuracy:
            print(f"Пропускаємо - результат вже задовільний")
            return True
        else:
            print(f"Запускаємо - потрібна краща точність")
            return False
    
    return False

def run_experiment(model_name, mode, learning_rate, num_epochs=10, skip_existing=True, min_accuracy=80.0):
    if skip_existing and should_skip_experiment(model_name, mode, min_accuracy):
        existing_result = find_existing_result(model_name, mode)
        return existing_result['accuracy'] if existing_result else 0.0
    
    print(f"\nЕКСПЕРИМЕНТ: {model_name.upper()} | {mode.upper()} | LR: {learning_rate}")
    
    trainer = TransferLearningTrainer(
        data_dir='data',
        batch_size=8,
        num_classes=3,
        image_size=64
    )
    
    try:
        trainer.setup_data()
        trainer.create_model(model_name, mode, learning_rate)
        trainer.train(num_epochs=num_epochs, patience=2)
        trainer.save_results(model_name, mode)
        
        return trainer.best_val_accuracy
        
    except Exception as e:
        print(f"Помилка в експерименті: {e}")
        return 0.0

if __name__ == '__main__':
    experiments = [
        {'model': 'resnet18', 'mode': 'feature_extraction', 'lr': 0.001},
        {'model': 'resnet18', 'mode': 'fine_tuning', 'lr': 0.0005},
        {'model': 'efficientnet_b0', 'mode': 'feature_extraction', 'lr': 0.001},
        {'model': 'efficientnet_b0', 'mode': 'fine_tuning', 'lr': 0.0005},
    ]
    
    print("=" * 70)
    print("ПЕРЕВІРКА ІСНУЮЧИХ РЕЗУЛЬТАТІВ")
    print("=" * 70)
    
    results = []
    for exp in experiments:
        print(f"\nПеревірка {exp['model']} - {exp['mode']}:")
        accuracy = run_experiment(
            exp['model'], 
            exp['mode'], 
            exp['lr'],
            skip_existing=True,
            min_accuracy=80.0
        )
        results.append({
            'model': exp['model'],
            'mode': exp['mode'], 
            'lr': exp['lr'],
            'accuracy': accuracy
        })
    
    print("\n" + "="*70)
    print("ПІДСУМКИ ЕКСПЕРИМЕНТІВ")
    print("="*70)
    
    successful_results = [r for r in results if r['accuracy'] > 0]
    
    for res in successful_results:
        status = "УСПІШНО" if res['accuracy'] >= 80.0 else "НЕУСПІШНО"
        print(f"{res['model']:15} | {res['mode']:18} | LR: {res['lr']:8} | Accuracy: {res['accuracy']:.2f}% | {status}")
    
    if successful_results:
        best_exp = max(successful_results, key=lambda x: x['accuracy'])
        print(f"\nНАЙКРАЩИЙ РЕЗУЛЬТАТ: {best_exp['model']} ({best_exp['mode']}) - {best_exp['accuracy']:.2f}%")
    else:
        print("\nВсі експерименти вже виконані з хорошою точністю!")