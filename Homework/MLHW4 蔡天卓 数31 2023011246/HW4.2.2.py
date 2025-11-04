import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split

# 加载数据
auto_mpg = fetch_ucirepo(id=9) 
X = auto_mpg.data.features 
y = auto_mpg.data.targets 

# 处理缺失值
X = X.dropna()
y = y.loc[X.index]

# 选择数值型特征
numerical_features = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin']
X = X[numerical_features]

# 转换为numpy数组
X = X.values
y = y.values.flatten()

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化函数
def standardize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std, mean, std

# 岭回归
class RidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        # 添加偏置项
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        
        # 岭回归闭式解: w = (X^T X + alpha * I)^(-1) X^T y
        n_features = X_with_bias.shape[1]
        identity_matrix = np.eye(n_features)
        # 不对偏置项进行正则化
        identity_matrix[0, 0] = 0
        
        XTX = X_with_bias.T @ X_with_bias
        regularization = self.alpha * identity_matrix
        XTy = X_with_bias.T @ y
        
        # 求解权重
        self.weights = np.linalg.inv(XTX + regularization) @ XTy
        self.bias = self.weights[0]
        self.weights = self.weights[1:]
    
    def predict(self, X):
        return X @ self.weights + self.bias

# 交叉验证
def cross_validate_ridge(X, y, alphas, k_folds=5):
    n_samples = X.shape[0]
    fold_size = n_samples // k_folds
    
    # 打乱数据
    indices = np.random.permutation(n_samples)
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    best_alpha = None
    best_mse = float('inf')
    
    for alpha in alphas:
        fold_mses = []
        
        for fold in range(k_folds):
            # 划分训练集和验证集
            val_start = fold * fold_size
            val_end = (fold + 1) * fold_size
            
            X_val = X_shuffled[val_start:val_end]
            y_val = y_shuffled[val_start:val_end]
            
            X_train_fold = np.concatenate([X_shuffled[:val_start], X_shuffled[val_end:]], axis=0)
            y_train_fold = np.concatenate([y_shuffled[:val_start], y_shuffled[val_end:]], axis=0)
            
            # 标准化
            X_train_scaled, mean, std = standardize(X_train_fold)
            X_val_scaled = (X_val - mean) / std
            
            # 训练模型
            model = RidgeRegression(alpha=alpha)
            model.fit(X_train_scaled, y_train_fold)
            
            # 预测并计算MSE
            y_pred = model.predict(X_val_scaled)
            mse = np.mean((y_pred - y_val) ** 2)
            fold_mses.append(mse)
        
        # 计算平均MSE
        avg_mse = np.mean(fold_mses)
        print(f"Alpha: {alpha}, Average MSE: {avg_mse:.4f}")
        
        if avg_mse < best_mse:
            best_mse = avg_mse
            best_alpha = alpha
    
    return best_alpha, best_mse

# 标准化训练数据
X_train_scaled, train_mean, train_std = standardize(X_train)
X_test_scaled = (X_test - train_mean) / train_std

# 定义要测试的alpha值
alphas = [0.01, 0.1, 1, 10, 100]

best_alpha, best_mse = cross_validate_ridge(X_train, y_train, alphas, k_folds=5)
print(f"\n最佳超参数 alpha: {best_alpha}")
print(f"交叉验证最佳MSE: {best_mse:.4f}")

# 使用最佳alpha在完整训练集上训练最终模型
final_model = RidgeRegression(alpha=best_alpha)
final_model.fit(X_train_scaled, y_train)

# 在测试集上评估
y_pred_test = final_model.predict(X_test_scaled)
test_mse = np.mean((y_pred_test - y_test) ** 2)
print(f"测试集MSE: {test_mse:.4f}")

# 计算R²分数
def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

r2 = r2_score(y_test, y_pred_test)
print(f"测试集R²分数: {r2:.4f}")

# 显示模型系数
print(f"\n模型系数: {final_model.weights}")
print(f"模型偏置: {final_model.bias:.4f}")