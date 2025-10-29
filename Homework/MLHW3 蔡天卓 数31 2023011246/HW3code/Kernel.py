import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import accuracy_score

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
cols = ["ID", "Diagnosis"] + [f"feature_{i}" for i in range(1,31)]
df = pd.read_csv(url, header=None, names=cols)

X = df.iloc[:, 2:].values
y = (df["Diagnosis"] == "M").astype(int).values
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
X_train, X_test = StandardScaler().fit_transform(X_train), StandardScaler().fit_transform(X_test)

# (1) RBF 核 SVM
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_rbf.fit(X_train, y_train)
print(f"[RBF SVM] Accuracy = {accuracy_score(y_test, svm_rbf.predict(X_test)):.4f}")

# (2) Gaussian Process Classifier
kernel = 1.0 * RBF(length_scale=1.0)
gpc = GaussianProcessClassifier(kernel=kernel, random_state=42)
gpc.fit(X_train, y_train)
print(f"[Gaussian Process Classifier] Accuracy = {accuracy_score(y_test, gpc.predict(X_test)):.4f}")
