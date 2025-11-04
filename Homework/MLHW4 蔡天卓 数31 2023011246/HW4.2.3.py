import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
bank_marketing = fetch_ucirepo(id=222) 

# data (as pandas dataframes) 
X = bank_marketing.data.features 
y = bank_marketing.data.targets 

# 数据基本信息
print("数据形状:")
print(f"特征矩阵形状: {X.shape}")
print(f"目标变量形状: {y.shape}")

print("\n目标变量分布:")
print(y.value_counts())

# 预处理：编码分类变量
print("\n开始数据预处理...")
label_encoders = {}
categorical_columns = X.select_dtypes(include=['object']).columns

for column in categorical_columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column].astype(str))
    label_encoders[column] = le
    print(f"已编码列: {column}, 类别数: {len(le.classes_)}")

# 编码目标变量
y_encoder = LabelEncoder()
y_encoded = y_encoder.fit_transform(y.values.ravel())
print(f"目标变量编码: {y_encoder.classes_}")

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"\n训练集大小: {X_train.shape}")
print(f"测试集大小: {X_test.shape}")

# 随机森林分类
print("\n=== 随机森林分类 ===")
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf = RandomForestClassifier(random_state=42)
rf_grid_search = GridSearchCV(rf, rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
rf_grid_search.fit(X_train, y_train)

best_rf = rf_grid_search.best_estimator_
y_pred_rf = best_rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

print(f"随机森林最佳参数: {rf_grid_search.best_params_}")
print(f"随机森林测试准确率: {accuracy_rf:.4f}")

# XGBoost分类
try:
    import xgboost as xgb
    
    print("\n=== XGBoost分类 ===")
    xgb_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.1, 0.01],
        'subsample': [0.8, 1.0]
    }
    
    xgb_model = xgb.XGBClassifier(random_state=42)
    xgb_grid_search = GridSearchCV(xgb_model, xgb_param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    xgb_grid_search.fit(X_train, y_train)
    
    best_xgb = xgb_grid_search.best_estimator_
    y_pred_xgb = best_xgb.predict(X_test)
    accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
    
    print(f"XGBoost最佳参数: {xgb_grid_search.best_params_}")
    print(f"XGBoost测试准确率: {accuracy_xgb:.4f}")
    
except ImportError:
    print("\nXGBoost未安装，跳过XGBoost部分")
    print("可以使用: pip install xgboost 安装")
    best_xgb = None

# 详细结果比较
print("\n" + "="*50)
print("模型性能比较")
print("="*50)

print(f"\n随机森林结果:")
print(f"最佳参数: {rf_grid_search.best_params_}")
print(f"准确率: {accuracy_rf:.4f}")
print("\n随机森林分类报告:")
print(classification_report(y_test, y_pred_rf, target_names=y_encoder.classes_))

if best_xgb is not None:
    print(f"\nXGBoost结果:")
    print(f"最佳参数: {xgb_grid_search.best_params_}")
    print(f"准确率: {accuracy_xgb:.4f}")
    print("\nXGBoost分类报告:")
    print(classification_report(y_test, y_pred_xgb, target_names=y_encoder.classes_))

# 绘制混淆矩阵
fig, axes = plt.subplots(1, 2 if best_xgb is not None else 1, figsize=(15, 6))

# 随机森林混淆矩阵
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', 
            xticklabels=y_encoder.classes_, 
            yticklabels=y_encoder.classes_, ax=axes[0] if best_xgb is not None else axes)
axes[0 if best_xgb is not None else axes].set_title('随机森林混淆矩阵')
axes[0 if best_xgb is not None else axes].set_xlabel('预测标签')
axes[0 if best_xgb is not None else axes].set_ylabel('真实标签')

if best_xgb is not None:
    # XGBoost混淆矩阵
    cm_xgb = confusion_matrix(y_test, y_pred_xgb)
    sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues', 
                xticklabels=y_encoder.classes_, 
                yticklabels=y_encoder.classes_, ax=axes[1])
    axes[1].set_title('XGBoost混淆矩阵')
    axes[1].set_xlabel('预测标签')
    axes[1].set_ylabel('真实标签')

plt.tight_layout()
plt.show()

# 特征重要性可视化
fig, axes = plt.subplots(1, 2 if best_xgb is not None else 1, figsize=(20, 8))

# 随机森林特征重要性
rf_importance = best_rf.feature_importances_
rf_indices = np.argsort(rf_importance)[::-1][:15]  # 取前15个重要特征

axes[0 if best_xgb is not None else axes].barh(range(len(rf_indices)), rf_importance[rf_indices])
axes[0 if best_xgb is not None else axes].set_yticks(range(len(rf_indices)))
axes[0 if best_xgb is not None else axes].set_yticklabels([X.columns[i] for i in rf_indices])
axes[0 if best_xgb is not None else axes].set_title('随机森林特征重要性 (前15个)')
axes[0 if best_xgb is not None else axes].set_xlabel('重要性')

if best_xgb is not None:
    # XGBoost特征重要性
    xgb_importance = best_xgb.feature_importances_
    xgb_indices = np.argsort(xgb_importance)[::-1][:15]  # 取前15个重要特征
    
    axes[1].barh(range(len(xgb_indices)), xgb_importance[xgb_indices])
    axes[1].set_yticks(range(len(xgb_indices)))
    axes[1].set_yticklabels([X.columns[i] for i in xgb_indices])
    axes[1].set_title('XGBoost特征重要性 (前15个)')
    axes[1].set_xlabel('重要性')

plt.tight_layout()
plt.show()

# 性能总结
print("\n" + "="*50)
print("性能总结")
print("="*50)

if best_xgb is not None:
    models_comparison = pd.DataFrame({
        'Model': ['Random Forest', 'XGBoost'],
        'Accuracy': [accuracy_rf, accuracy_xgb],
        'Best Parameters': [rf_grid_search.best_params_, xgb_grid_search.best_params_]
    })
else:
    models_comparison = pd.DataFrame({
        'Model': ['Random Forest'],
        'Accuracy': [accuracy_rf],
        'Best Parameters': [rf_grid_search.best_params_]
    })

print(models_comparison)

if best_xgb is not None:
    if accuracy_rf > accuracy_xgb:
        print(f"\n随机森林表现更好，准确率高出 {accuracy_rf - accuracy_xgb:.4f}")
    elif accuracy_xgb > accuracy_rf:
        print(f"\nXGBoost表现更好，准确率高出 {accuracy_xgb - accuracy_rf:.4f}")
    else:
        print(f"\n两个模型表现相同")