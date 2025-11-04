import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# 加载数据
bank_marketing = fetch_ucirepo(id=222) 
X = bank_marketing.data.features 
y = bank_marketing.data.targets 

# 预处理：编码分类变量
label_encoders = {}
categorical_columns = X.select_dtypes(include=['object']).columns

for column in categorical_columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column].astype(str))
    label_encoders[column] = le

# 编码目标变量
y_encoder = LabelEncoder()
y_encoded = y_encoder.fit_transform(y.values.ravel())

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# 随机森林分类
print("\n=== Random Forest ===")
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5]
}

rf = RandomForestClassifier(random_state=42)
rf_grid_search = GridSearchCV(rf, rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
rf_grid_search.fit(X_train, y_train)

best_rf = rf_grid_search.best_estimator_
y_pred_rf = best_rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

print(f"Best parameters: {rf_grid_search.best_params_}")
print(f"Accuracy: {accuracy_rf:.4f}")

# XGBoost分类
try:
    import xgboost as xgb
    
    print("\n=== XGBoost ===")
    xgb_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 6],
        'learning_rate': [0.1, 0.01]
    }
    
    xgb_model = xgb.XGBClassifier(random_state=42)
    xgb_grid_search = GridSearchCV(xgb_model, xgb_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    xgb_grid_search.fit(X_train, y_train)
    
    best_xgb = xgb_grid_search.best_estimator_
    y_pred_xgb = best_xgb.predict(X_test)
    accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
    
    print(f"Best parameters: {xgb_grid_search.best_params_}")
    print(f"Accuracy: {accuracy_xgb:.4f}")
    
except ImportError:
    print("\nXGBoost not installed, skipping XGBoost part")
    print("Install with: pip install xgboost")
    best_xgb = None

# 结果比较
print("\n=== Model Comparison ===")
if best_xgb is not None:
    models_comparison = pd.DataFrame({
        'Model': ['Random Forest', 'XGBoost'],
        'Accuracy': [accuracy_rf, accuracy_xgb]
    })
else:
    models_comparison = pd.DataFrame({
        'Model': ['Random Forest'],
        'Accuracy': [accuracy_rf]
    })

print(models_comparison)

# 特征重要性可视化
plt.figure(figsize=(10, 6))
rf_importance = best_rf.feature_importances_
rf_indices = np.argsort(rf_importance)[::-1][:10]  # Top 10 features

plt.barh(range(len(rf_indices)), rf_importance[rf_indices])
plt.yticks(range(len(rf_indices)), [X.columns[i] for i in rf_indices])
plt.title('Random Forest Feature Importance (Top 10)')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()

# 打印简要分类报告
print("\n=== Random Forest Classification Report ===")
print(classification_report(y_test, y_pred_rf, target_names=y_encoder.classes_))

if best_xgb is not None:
    print("\n=== XGBoost Classification Report ===")
    print(classification_report(y_test, y_pred_xgb, target_names=y_encoder.classes_))