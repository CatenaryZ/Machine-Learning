import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
cols = ["ID", "Diagnosis"] + [f"feature_{i}" for i in range(1,31)]
df = pd.read_csv(url, header=None, names=cols)

X = df.iloc[:, 2:].values
y = (df["Diagnosis"] == "M").astype(int).values
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
X_train = StandardScaler().fit_transform(X_train)
X_test  = StandardScaler().fit_transform(X_test)

lda = LinearDiscriminantAnalysis().fit(X_train, y_train)
qda = QuadraticDiscriminantAnalysis().fit(X_train, y_train)
gnb = GaussianNB().fit(X_train, y_train)

print(f"[LDA] Accuracy = {accuracy_score(y_test, lda.predict(X_test)):.4f}")
print(f"[QDA] Accuracy = {accuracy_score(y_test, qda.predict(X_test)):.4f}")
print(f"[GaussianNB] Accuracy = {accuracy_score(y_test, gnb.predict(X_test)):.4f}")
