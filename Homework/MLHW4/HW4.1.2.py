import numpy as np
from ucimlrepo import fetch_ucirepo 

# 1. 加载葡萄酒数据集
wine = fetch_ucirepo(id=109) 
X = wine.data.features 
y = wine.data.targets 

print(f"数据集形状: {X.shape}")

# 2. 数据标准化
X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# 3. 计算协方差矩阵
cov_matrix = np.cov(X_std, rowvar=False)

# 4. 计算特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# 5. 按特征值大小排序
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# 6. 提取前三个主成分和对应的方差
pc1 = eigenvectors[:, 0]
pc2 = eigenvectors[:, 1]
pc3 = eigenvectors[:, 2]

variance_pc1 = eigenvalues[0]
variance_pc2 = eigenvalues[1]
variance_pc3 = eigenvalues[2]

# 7. 输出结果
print(f"\n前三个主成分的方差:")
print(f"PC1: {variance_pc1:.4f}")
print(f"PC2: {variance_pc2:.4f}")
print(f"PC3: {variance_pc3:.4f}")

# 8. 输出前三个主成分向量
print(f"\n第一主成分向量 (PC1):")
print(pc1)

print(f"\n第二主成分向量 (PC2):")
print(pc2)

print(f"\n第三主成分向量 (PC3):")
print(pc3)