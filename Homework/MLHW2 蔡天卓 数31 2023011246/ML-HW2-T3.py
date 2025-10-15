import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 设置随机种子
# torch.manual_seed(5)

class ThreeLayerNN(nn.Module):
    def __init__(self):
        super(ThreeLayerNN, self).__init__()
        # 输入层到隐藏层: 3个输入节点(b_h, x1, x2) -> 2个隐藏节点(h1, h2)
        # 注意: b_o是隐藏层的偏置，不在此连接中
        self.input_to_hidden = nn.Linear(3, 2)  # 3输入 -> 2输出
        
        # 隐藏层到输出层: 3个隐藏节点(b_o, h1, h2) -> 3个输出节点(y1, y2, y3)
        self.hidden_to_output = nn.Linear(3, 3)  # 3输入 -> 3输出
        
        # 激活函数 - 根据你的网络图选择合适的激活函数
        self.activation = nn.Sigmoid() 
    
    def forward(self, x):
        # x的形状: [batch_size, 2] - 只有x1, x2，b_h是常数1
        # 添加偏置节点b_h (值为1)
        x_with_bias = torch.cat([torch.ones(x.size(0), 1), x], dim=1)
        
        # 输入层到隐藏层
        hidden_input = self.input_to_hidden(x_with_bias)  # 输出h1, h2
        hidden_output = self.activation(hidden_input)
        
        # 添加隐藏层偏置节点b_o (值为1)
        hidden_with_bias = torch.cat([
            torch.ones(hidden_output.size(0), 1), 
            hidden_output
        ], dim=1)
        
        # 隐藏层到输出层
        output = self.hidden_to_output(hidden_with_bias)
        return output

# 初始化模型
model = ThreeLayerNN()

# 设置学习率
learning_rate = 0.001

print("设置初始权重...")
with torch.no_grad():
    # 输入层到隐藏层的权重 (2x3矩阵)
    # 格式: [[w_bh_h1, w_x1_h1, w_x2_h1], 
    #        [w_bh_h2, w_x1_h2, w_x2_h2]]
    model.input_to_hidden.weight.data = torch.tensor([
        [0.1, 0.2, 0.3],  # h1的权重
        [0.4, 0.5, 0.6]   # h2的权重
    ], dtype=torch.float32)
    
    # 输入层到隐藏层的偏置 (通常设为0，因为我们已经有了b_h)
    model.input_to_hidden.bias.data = torch.tensor([0.0, 0.0], dtype=torch.float32)
    
    # 隐藏层到输出层的权重 (3x3矩阵)
    # 格式: [[w_bo_y1, w_h1_y1, w_h2_y1],
    #        [w_bo_y2, w_h1_y2, w_h2_y2], 
    #        [w_bo_y3, w_h1_y3, w_h2_y3]]
    model.hidden_to_output.weight.data = torch.tensor([
        [0.7, 0.8, 0.9],    # y1的权重
        [1.0, 1.1, 1.2],    # y2的权重  
        [1.3, 1.4, 1.5]     # y3的权重
    ], dtype=torch.float32)
    
    # 隐藏层到输出层的偏置 (通常设为0，因为我们已经有了b_o)
    model.hidden_to_output.bias.data = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

print("初始权重设置完成!")
print("输入层->隐藏层权重:\n", model.input_to_hidden.weight.data)
print("隐藏层->输出层权重:\n", model.hidden_to_output.weight.data)

# 生成随机数据点 (4个样本)
# 每个样本有2个特征: x1, x2 (b_h是常数1，不需要生成)
X = torch.randn(4, 2)  # 4个样本，每个样本有x1, x2
y = torch.randn(4, 3)  # 4个目标值，每个有y1, y2, y3

print("\n训练数据:")
print("输入 X (x1, x2):\n", X)
print("目标 y (y1, y2, y3):\n", y)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 训练循环 - 5个步骤
print("\n开始训练...")
for step in range(5):
    # 前向传播
    outputs = model(X)
    loss = criterion(outputs, y)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 打印每一步的结果
    print(f"\n=== 步骤 {step + 1} ===")
    print(f"损失: {loss.item():.6f}")
    print("输入层->隐藏层权重:")
    print(model.input_to_hidden.weight.data)
    print("隐藏层->输出层权重:")
    print(model.hidden_to_output.weight.data)
    
    # 可以打印梯度
    if step == 0:
        print("\n第一次迭代的梯度:")
        print("输入层->隐藏层权重梯度:")
        print(model.input_to_hidden.weight.grad)
        print("隐藏层->输出层权重梯度:")
        print(model.hidden_to_output.weight.grad)

print("\n训练完成!")

# 打印最终预测结果
with torch.no_grad():
    final_outputs = model(X)
    print("\n最终预测结果:")
    print(final_outputs)
    print("\n目标值:")
    print(y)