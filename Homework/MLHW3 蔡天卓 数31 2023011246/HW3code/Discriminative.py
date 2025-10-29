import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Perceptron
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 载入数据
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
cols = ["ID", "Diagnosis"] + [f"feature_{i}" for i in range(1,31)]
df = pd.read_csv(url, header=None, names=cols)

X = df.iloc[:, 2:].values
y = (df["Diagnosis"] == "M").astype(int).values

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

# 1. 线性回归 (MSE 判别)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = (lr.predict(X_test) > 0.5).astype(int)
print(f"[Linear Regression] Accuracy = {accuracy_score(y_test, y_pred_lr):.4f}")

# 2. 感知器
ppn = Perceptron(max_iter=1000, tol=1e-3, random_state=42)
ppn.fit(X_train, y_train)
print(f"[Perceptron] Accuracy = {accuracy_score(y_test, ppn.predict(X_test)):.4f}")

# 3. 线性 SVM
svm_lin = SVC(kernel='linear', C=1.0, random_state=42)
svm_lin.fit(X_train, y_train)
print(f"[Linear SVM] Accuracy = {accuracy_score(y_test, svm_lin.predict(X_test)):.4f}")