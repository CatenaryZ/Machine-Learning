import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
cols = ["ID", "Diagnosis"] + [f"feature_{i}" for i in range(1,31)]
df = pd.read_csv(url, header=None, names=cols)

X = df.iloc[:, 2:].values
y = (df["Diagnosis"] == "M").astype(int).values
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
X_train, X_test = StandardScaler().fit_transform(X_train), StandardScaler().fit_transform(X_test)

# Logistic Regression
logreg = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
logreg.fit(X_train, y_train)
print(f"[Logistic Regression] Accuracy = {accuracy_score(y_test, logreg.predict(X_test)):.4f}")

# Neural Network
mlp = MLPClassifier(hidden_layer_sizes=(50,25), activation='relu', solver='adam', max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)
print(f"[Neural Network MLP] Accuracy = {accuracy_score(y_test, mlp.predict(X_test)):.4f}")
