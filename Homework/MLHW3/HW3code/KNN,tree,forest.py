import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
cols = ["ID", "Diagnosis"] + [f"feature_{i}" for i in range(1,31)]
df = pd.read_csv(url, header=None, names=cols)

X = df.iloc[:, 2:].values
y = (df["Diagnosis"] == "M").astype(int).values
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
X_train, X_test = StandardScaler().fit_transform(X_train), StandardScaler().fit_transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
dt  = DecisionTreeClassifier(max_depth=5, random_state=42)
rf  = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

knn.fit(X_train, y_train)
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)

print(f"[KNN] Accuracy = {accuracy_score(y_test, knn.predict(X_test)):.4f}")
print(f"[Decision Tree] Accuracy = {accuracy_score(y_test, dt.predict(X_test)):.4f}")
print(f"[Random Forest] Accuracy = {accuracy_score(y_test, rf.predict(X_test)):.4f}")
