import numpy as np

np.random.seed(42)

# 网络结构
input_size = 2
hidden_size = 2
output_size = 3
learning_rate = 0.001
n_epochs = 5
n_samples = 4

# 1. 随机生成数据
X = np.random.randn(n_samples, input_size)  # 4x2
t = np.eye(output_size)[np.random.choice(output_size, n_samples)]  # 4x3 的 one-hot 标签

# 2. 初始化权重
W1 = np.random.randn(hidden_size, input_size) * 0.01
b1 = np.zeros(hidden_size)
W2 = np.random.randn(output_size, hidden_size) * 0.01
b2 = np.zeros(output_size)

print("Initial weights:")
print("W1 (hidden weights):", W1)
print("b1 (hidden biases):", b1)
print("W2 (output weights):", W2)
print("b2 (output biases):", b2)
print()

# 3. 训练循环
for epoch in range(n_epochs):
    # 前向传播
    # 隐藏层
    a1 = X @ W1.T + b1  # (4,2) @ (2,2).T -> (4,2) + (2,) -> (4,2)
    z1 = 1 / (1 + np.exp(-a1))  # sigmoid
    
    # 输出层
    a2 = z1 @ W2.T + b2  # (4,2) @ (3,2).T -> (4,3) + (3,) -> (4,3)
    # softmax
    exp_scores = np.exp(a2 - np.max(a2, axis=1, keepdims=True))
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # (4,3)
    
    # 损失
    loss = -np.sum(t * np.log(probs + 1e-8)) / n_samples
    print(f"Epoch {epoch+1}, Loss: {loss:.6f}")
    
    # 反向传播
    # 输出层误差
    delta_a2 = (probs - t) / n_samples  # (4,3)
    
    # 隐藏层误差
    delta_a1 = (delta_a2 @ W2) * (z1 * (1 - z1))  # (4,3) @ (3,2) -> (4,2) * (4,2) -> (4,2)
    
    # 梯度
    dW2 = delta_a2.T @ z1  # (3,4) @ (4,2) -> (3,2)
    db2 = np.sum(delta_a2, axis=0)  # (3,)
    dW1 = delta_a1.T @ X  # (2,4) @ (4,2) -> (2,2)
    db1 = np.sum(delta_a1, axis=0)  # (2,)
    
    # 更新权重
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

print("\nFinal weights:")
print("W1:", W1)
print("b1:", b1)
print("W2:", W2)
print("b2:", b2)