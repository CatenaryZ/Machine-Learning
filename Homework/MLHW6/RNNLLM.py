import torch
import torch.nn as nn
import numpy as np
import random
from collections import Counter
import time


import os

# 打印当前工作目录
print(f"当前工作目录: {os.getcwd()}")

# 检查当前目录下的文件
print("当前目录下的文件:")
for file in os.listdir():
    print(f"  - {file}")

# 检查是否有 input.txt
if 'input.txt' in os.listdir():
    print("在当前目录找到 input.txt")
else:
    print("在当前目录没有找到 input.txt")

# 读取数据
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(f"文本长度: {len(text)} 字符")
print(f"前100个字符: {text[:100]}")

# 创建字符到索引的映射
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"词汇表大小: {vocab_size}")
print(f"字符集: {''.join(chars)}")

char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# 将文本转换为索引序列
data = [char_to_idx[ch] for ch in text]

# 数据准备
def create_dataset(data, seq_length=100):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return torch.tensor(X), torch.tensor(y)

seq_length = 100
X, y = create_dataset(data, seq_length)
print(f"训练样本数: {len(X)}")

# 划分训练集和验证集
split_idx = int(0.9 * len(X))
X_train, y_train = X[:split_idx], y[:split_idx]
X_val, y_val = X[split_idx:], y[split_idx:]

print(f"训练集大小: {len(X_train)}")
print(f"验证集大小: {len(y_val)}")

# 批量数据加载器
def batch_generator(X, y, batch_size=64, shuffle=True):
    indices = list(range(len(X)))
    if shuffle:
        random.shuffle(indices)
    
    for start_idx in range(0, len(indices), batch_size):
        batch_indices = indices[start_idx:start_idx+batch_size]
        batch_X = X[batch_indices]
        batch_y = y[batch_indices]
        yield batch_X, batch_y

# GRU语言模型
class CharGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, n_layers=2, dropout=0.2):
        super(CharGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, n_layers, 
                         batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        
        if hidden is None:
            hidden = self.init_hidden(batch_size)
        
        embedded = self.embedding(x)
        output, hidden = self.gru(embedded, hidden)
        output = self.dropout(output)
        
        # 只取最后一个时间步的输出
        output = output[:, -1, :]
        output = self.fc(output)
        
        return output, hidden
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()
        return hidden
    
    def generate(self, start_str, length=500, temperature=0.8, device='cpu'):
        self.eval()
        
        # 初始化隐藏状态
        hidden = None
        
        # 将起始字符串转换为索引
        input_seq = torch.tensor([[char_to_idx[ch] for ch in start_str]], device=device)
        
        # 前向传播获取初始隐藏状态
        with torch.no_grad():
            embedded = self.embedding(input_seq)
            _, hidden = self.gru(embedded, hidden)
        
        # 生成文本
        generated = list(start_str)
        input_char = input_seq[:, -1:]
        
        for _ in range(length):
            with torch.no_grad():
                embedded = self.embedding(input_char)
                output, hidden = self.gru(embedded, hidden)
                output = self.fc(output[:, -1, :])
                
                # 应用温度采样
                output = output / temperature
                probs = torch.softmax(output, dim=-1)
                
                # 采样下一个字符
                next_char_idx = torch.multinomial(probs, 1).item()
                
                # 添加到生成文本中
                generated.append(idx_to_char[next_char_idx])
                
                # 准备下一个输入
                input_char = torch.tensor([[next_char_idx]], device=device)
        
        return ''.join(generated)

# 训练函数
def train_model(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=64, lr=0.001, device='cpu'):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        batch_count = 0
        
        # 训练阶段
        for batch_X, batch_y in batch_generator(X_train, y_train, batch_size):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            hidden = None
            
            # 前向传播
            output, hidden = model(batch_X, hidden)
            loss = criterion(output, batch_y)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        avg_train_loss = epoch_loss / batch_count
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch_X, batch_y in batch_generator(X_val, y_val, batch_size, shuffle=False):
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                hidden = None
                
                output, hidden = model(batch_X, hidden)
                loss = criterion(output, batch_y)
                
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        # 打印进度
        if (epoch + 1) % 1 == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss: {avg_val_loss:.4f}")
            
            # 生成示例文本
            with torch.no_grad():
                start_str = "First Citizen:"
                if len(start_str) > seq_length:
                    start_str = start_str[:seq_length]
                
                generated = model.generate(
                    start_str, 
                    length=200, 
                    temperature=0.8,
                    device=device
                )
                print(f"  生成文本示例:\n{generated[:500]}\n")
    
    return train_losses, val_losses

# 创建模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

model = CharGRU(
    vocab_size=vocab_size,
    embedding_dim=128,
    hidden_dim=256,
    n_layers=2,
    dropout=0.2
)

print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

# 训练模型
print("开始训练...")
train_losses, val_losses = train_model(
    model, 
    X_train, y_train, 
    X_val, y_val,
    epochs=10,  # 可以增加更多epoch以获得更好效果
    batch_size=64,
    lr=0.001,
    device=device
)

# 保存模型
torch.save({
    'model_state_dict': model.state_dict(),
    'char_to_idx': char_to_idx,
    'idx_to_char': idx_to_char,
    'vocab_size': vocab_size,
    'seq_length': seq_length,
}, 'char_gru_model.pth')

print("模型已保存到 char_gru_model.pth")

# 生成更多文本示例
def generate_text_samples(model, start_strings, lengths=[200, 300, 500], temperatures=[0.5, 0.8, 1.0]):
    model.eval()
    
    for start_str in start_strings:
        print(f"\n{'='*60}")
        print(f"起始文本: '{start_str}'")
        print(f"{'='*60}")
        
        for length in lengths:
            for temp in temperatures:
                generated = model.generate(
                    start_str[:seq_length],
                    length=length,
                    temperature=temp,
                    device=device
                )
                
                print(f"\n长度: {length}, 温度: {temp}")
                print(f"{'-'*40}")
                print(generated)
                print(f"{'-'*40}\n")

# 测试不同的起始文本
start_strings = [
    "First Citizen:",
    "MENENIUS:",
    "CORIOLANUS:",
    "The gods",
    "I pray you",
    "To be or not to be",  # 莎士比亚其他作品的经典台词
]

generate_text_samples(model, start_strings)

# 评估困惑度（Perplexity）
def evaluate_perplexity(model, X, y, batch_size=64, device='cpu'):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch_X, batch_y in batch_generator(X, y, batch_size, shuffle=False):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            hidden = None
            
            output, hidden = model(batch_X, hidden)
            loss = criterion(output, batch_y)
            
            total_loss += loss.item() * len(batch_y)
            total_samples += len(batch_y)
    
    avg_loss = total_loss / total_samples
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return perplexity, avg_loss

# 计算困惑度
train_perplexity, train_loss = evaluate_perplexity(model, X_train, y_train, device=device)
val_perplexity, val_loss = evaluate_perplexity(model, X_val, y_val, device=device)

print(f"\n{'='*60}")
print("模型评估结果:")
print(f"{'='*60}")
print(f"训练集 - 损失: {train_loss:.4f}, 困惑度: {train_perplexity:.2f}")
print(f"验证集 - 损失: {val_loss:.4f}, 困惑度: {val_perplexity:.2f}")

# 加载模型并生成文本的函数
def load_and_generate(model_path, start_str, length=500, temperature=0.8):
    checkpoint = torch.load(model_path, map_location=device)
    
    loaded_model = CharGRU(
        vocab_size=checkpoint['vocab_size'],
        embedding_dim=128,
        hidden_dim=256,
        n_layers=2,
        dropout=0.2
    )
    
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.to(device)
    
    # 设置全局变量（为了generate函数能访问）
    global char_to_idx, idx_to_char
    char_to_idx = checkpoint['char_to_idx']
    idx_to_char = checkpoint['idx_to_char']
    
    # 生成文本
    loaded_model.eval()
    generated = loaded_model.generate(
        start_str,
        length=length,
        temperature=temperature,
        device=device
    )
    
    return generated

# 演示加载模型并生成
print("\n演示加载模型并生成文本...")
try:
    generated_text = load_and_generate(
        'char_gru_model.pth',
        "First Citizen: We are resolved",
        length=300,
        temperature=0.7
    )
    print(f"生成的文本:\n{generated_text}")
except Exception as e:
    print(f"加载模型时出错: {e}")