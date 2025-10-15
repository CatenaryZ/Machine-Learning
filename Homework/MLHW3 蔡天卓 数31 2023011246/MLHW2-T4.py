import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import time

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class CIFAR10Classifier:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # 加载数据集
        self.train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=self.transform)
        self.test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=self.transform)
        
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck')
    
    def show_sample_images(self):
        """显示一些样本图像"""
        fig, axes = plt.subplots(2, 5, figsize=(12, 6))
        for i in range(10):
            img, label = self.train_dataset[i]
            img = img / 2 + 0.5  # 反标准化
            npimg = img.numpy()
            ax = axes[i//5, i%5]
            ax.imshow(np.transpose(npimg, (1, 2, 0)))
            ax.set_title(self.classes[label])
            ax.axis('off')
        plt.tight_layout()
        plt.show()

class LogisticRegression(nn.Module):
    """逻辑回归模型（单层网络）"""
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        return self.linear(x)

class NeuralNetwork(nn.Module):
    """神经网络模型"""
    def __init__(self, input_size, hidden_sizes, num_classes, activation='relu'):
        super(NeuralNetwork, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # 构建隐藏层
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            
            # 选择激活函数
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.1))
            
            layers.append(nn.Dropout(0.3))  # 添加dropout防止过拟合
            prev_size = hidden_size
        
        # 输出层
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class ModelTrainer:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.test_accuracies = []
    
    def train(self, train_loader, test_loader, criterion, optimizer, num_epochs):
        print(f"Training {self.model.__class__.__name__}...")
        
        for epoch in range(num_epochs):
            # 训练阶段
            self.model.train()
            running_loss = 0.0
            for i, (images, labels) in enumerate(train_loader):
                images = images.reshape(images.size(0), -1).to(self.device)
                labels = labels.to(self.device)
                
                # 前向传播
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            epoch_loss = running_loss / len(train_loader)
            self.train_losses.append(epoch_loss)
            
            # 测试阶段
            test_acc = self.evaluate(test_loader)
            self.test_accuracies.append(test_acc)
            
            if (epoch + 1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Test Acc: {test_acc:.2f}%')
    
    def evaluate(self, test_loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.reshape(images.size(0), -1).to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return 100 * correct / total

def hyperparameter_tuning():
    """超参数调优实验"""
    cifar = CIFAR10Classifier()
    input_size = 3 * 32 * 32  # CIFAR-10图像尺寸
    num_classes = 10
    
    # 定义不同的超参数组合
    experiments = [
        # 网络架构实验
        {
            'name': 'Logistic Regression',
            'model_type': 'logistic',
            'hidden_sizes': [],
            'batch_size': 128,
            'learning_rate': 0.01,
            'num_epochs': 50,
            'activation': 'none'
        },
        {
            'name': '1-Layer NN (ReLU)',
            'model_type': 'neural_net',
            'hidden_sizes': [512],
            'batch_size': 64,
            'learning_rate': 0.001,
            'num_epochs': 50,
            'activation': 'relu'
        },
        {
            'name': '2-Layer NN (ReLU)',
            'model_type': 'neural_net',
            'hidden_sizes': [512, 256],
            'batch_size': 64,
            'learning_rate': 0.001,
            'num_epochs': 50,
            'activation': 'relu'
        },
        {
            'name': '3-Layer NN (ReLU)',
            'model_type': 'neural_net',
            'hidden_sizes': [512, 256, 128],
            'batch_size': 64,
            'learning_rate': 0.001,
            'num_epochs': 50,
            'activation': 'relu'
        },
        # 激活函数实验
        {
            'name': '2-Layer NN (Sigmoid)',
            'model_type': 'neural_net',
            'hidden_sizes': [512, 256],
            'batch_size': 64,
            'learning_rate': 0.001,
            'num_epochs': 50,
            'activation': 'sigmoid'
        },
        {
            'name': '2-Layer NN (Tanh)',
            'model_type': 'neural_net',
            'hidden_sizes': [512, 256],
            'batch_size': 64,
            'learning_rate': 0.001,
            'num_epochs': 50,
            'activation': 'tanh'
        },
        # 批大小实验
        {
            'name': 'Batch Size 32',
            'model_type': 'neural_net',
            'hidden_sizes': [512, 256],
            'batch_size': 32,
            'learning_rate': 0.001,
            'num_epochs': 50,
            'activation': 'relu'
        },
        {
            'name': 'Batch Size 128',
            'model_type': 'neural_net',
            'hidden_sizes': [512, 256],
            'batch_size': 128,
            'learning_rate': 0.001,
            'num_epochs': 50,
            'activation': 'relu'
        },
        # 学习率实验
        {
            'name': 'Learning Rate 0.01',
            'model_type': 'neural_net',
            'hidden_sizes': [512, 256],
            'batch_size': 64,
            'learning_rate': 0.01,
            'num_epochs': 50,
            'activation': 'relu'
        },
        {
            'name': 'Learning Rate 0.0001',
            'model_type': 'neural_net',
            'hidden_sizes': [512, 256],
            'batch_size': 64,
            'learning_rate': 0.0001,
            'num_epochs': 50,
            'activation': 'relu'
        }
    ]
    
    results = []
    
    for exp in experiments:
        print(f"\n{'='*50}")
        print(f"Experiment: {exp['name']}")
        print(f"{'='*50}")
        
        # 创建数据加载器
        train_loader = DataLoader(cifar.train_dataset, batch_size=exp['batch_size'], shuffle=True)
        test_loader = DataLoader(cifar.test_dataset, batch_size=exp['batch_size'], shuffle=False)
        
        # 创建模型
        if exp['model_type'] == 'logistic':
            model = LogisticRegression(input_size, num_classes)
        else:
            model = NeuralNetwork(input_size, exp['hidden_sizes'], num_classes, exp['activation'])
        
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=exp['learning_rate'])
        
        # 训练模型
        trainer = ModelTrainer(model, device)
        start_time = time.time()
        trainer.train(train_loader, test_loader, criterion, optimizer, exp['num_epochs'])
        training_time = time.time() - start_time
        
        # 最终评估
        final_accuracy = trainer.evaluate(test_loader)
        
        results.append({
            'name': exp['name'],
            'final_accuracy': final_accuracy,
            'training_time': training_time,
            'train_losses': trainer.train_losses,
            'test_accuracies': trainer.test_accuracies,
            'parameters': exp
        })
        
        print(f"Final Test Accuracy: {final_accuracy:.2f}%")
        print(f"Training Time: {training_time:.2f} seconds")
    
    return results

def plot_results(results):
    """绘制结果图表"""
    # 准确率比较
    names = [r['name'] for r in results]
    accuracies = [r['final_accuracy'] for r in results]
    
    plt.figure(figsize=(15, 10))
    
    # 子图1：最终准确率比较
    plt.subplot(2, 2, 1)
    bars = plt.bar(names, accuracies, color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Final Test Accuracy Comparison')
    
    # 在柱状图上添加数值
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{acc:.1f}%', ha='center', va='bottom')
    
    # 子图2：训练损失曲线
    plt.subplot(2, 2, 2)
    for result in results[:4]:  # 只显示前4个模型的损失曲线
        plt.plot(result['train_losses'], label=result['name'])
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Curves')
    plt.legend()
    plt.yscale('log')  # 使用对数坐标更好地观察损失下降
    
    # 子图3：测试准确率曲线
    plt.subplot(2, 2, 3)
    for result in results[:4]:  # 只显示前4个模型的准确率曲线
        plt.plot(result['test_accuracies'], label=result['name'])
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy Curves')
    plt.legend()
    
    # 子图4：训练时间比较
    plt.subplot(2, 2, 4)
    times = [r['training_time'] for r in results]
    bars = plt.bar(names, times, color='lightcoral')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time Comparison')
    
    # 在柱状图上添加数值
    for bar, time_val in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{time_val:.1f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def summarize_findings(results):
    """总结实验结果"""
    print("\n" + "="*80)
    print("SUMMARY OF FINDINGS")
    print("="*80)
    
    # 找到最佳模型
    best_model = max(results, key=lambda x: x['final_accuracy'])
    
    print(f"\n🏆 BEST MODEL: {best_model['name']}")
    print(f"   Final Accuracy: {best_model['final_accuracy']:.2f}%")
    print(f"   Training Time: {best_model['training_time']:.2f} seconds")
    
    print("\n📊 KEY INSIGHTS:")
    
    # 网络架构影响
    arch_results = [r for r in results if 'Layer' in r['name'] and 'ReLU' in r['name']]
    arch_results.sort(key=lambda x: len(x['parameters']['hidden_sizes']))
    
    print("\n1. NETWORK ARCHITECTURE IMPACT:")
    for arch in arch_results:
        layers = len(arch['parameters']['hidden_sizes'])
        print(f"   - {layers}-layer NN: {arch['final_accuracy']:.2f}%")
    
    # 激活函数影响
    activation_results = [r for r in results if '2-Layer NN' in r['name'] and '(' in r['name']]
    print("\n2. ACTIVATION FUNCTION IMPACT:")
    for act in activation_results:
        activation = act['parameters']['activation']
        print(f"   - {activation}: {act['final_accuracy']:.2f}%")
    
    # 批大小影响
    batch_results = [r for r in results if 'Batch Size' in r['name']]
    print("\n3. BATCH SIZE IMPACT:")
    for batch in batch_results:
        size = batch['parameters']['batch_size']
        print(f"   - Batch Size {size}: {batch['final_accuracy']:.2f}%")
    
    # 学习率影响
    lr_results = [r for r in results if 'Learning Rate' in r['name']]
    print("\n4. LEARNING RATE IMPACT:")
    for lr in lr_results:
        rate = lr['parameters']['learning_rate']
        print(f"   - Learning Rate {rate}: {lr['final_accuracy']:.2f}%")
    
    # 逻辑回归 vs 神经网络
    lr_result = [r for r in results if r['name'] == 'Logistic Regression'][0]
    best_nn = max([r for r in results if r['name'] != 'Logistic Regression'], key=lambda x: x['final_accuracy'])
    
    print(f"\n5. LOGISTIC REGRESSION vs NEURAL NETWORK:")
    print(f"   - Logistic Regression: {lr_result['final_accuracy']:.2f}%")
    print(f"   - Best Neural Network: {best_nn['final_accuracy']:.2f}%")
    print(f"   - Improvement: {best_nn['final_accuracy'] - lr_result['final_accuracy']:.2f}%")

if __name__ == "__main__":
    # 显示样本图像
    cifar = CIFAR10Classifier()
    cifar.show_sample_images()
    
    # 进行超参数调优实验
    print("Starting Hyperparameter Tuning Experiments...")
    results = hyperparameter_tuning()
    
    # 绘制结果
    plot_results(results)
    
    # 总结发现
    summarize_findings(results)