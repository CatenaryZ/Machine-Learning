# === 导入必要的包 ===
from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# === 1. 获取数据 ===
online_retail = fetch_ucirepo(id=352)
X = online_retail.data.features

# === 2. 数据清洗与特征选择 ===
X = X.dropna(subset=['CustomerID'])
X['CustomerID'] = X['CustomerID'].astype(int)

# 聚合为每个顾客的数据
customer_df = X.groupby('CustomerID').agg({
    'Quantity': 'sum',           # 总购买量
    'UnitPrice': 'mean'          # 平均单价
}).reset_index()

# 增加总消费额特征
customer_df['TotalSpend'] = X.groupby('CustomerID').apply(
    lambda df: np.sum(df['Quantity'] * df['UnitPrice'])
).values

# 仅使用数值特征
features = customer_df[['Quantity', 'UnitPrice', 'TotalSpend']]

# === 3. 数据标准化 ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# === 4. 肘部法则曲线 ===
inertias = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(K_range, inertias, marker='o')
plt.xlabel('K')
plt.ylabel('Inertia')
plt.title('Elbow Method for K Selection')
plt.show()

# === 5. 轮廓系数分析 ===
silhouette_scores = []
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)

plt.figure(figsize=(6, 4))
plt.plot(K_range, silhouette_scores, marker='o', color='orange')
plt.xlabel('K')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis for K Selection')
plt.show()

# === 计算聚类 ===
k_final = 4
kmeans_final = KMeans(n_clusters=k_final, random_state=42)
customer_df['Cluster'] = kmeans_final.fit_predict(X_scaled)

# === 7. 输出 K=4 的肘部法则结果（Inertia） ===
inertia_k4 = kmeans_final.inertia_
print(f"当 K={k_final} 时的 Inertia（肘部法则指标）为: {inertia_k4:.2f}")

# === 8. 输出每个簇的平均特征 ===
cluster_summary = customer_df.groupby('Cluster')[['Quantity', 'UnitPrice', 'TotalSpend']].mean()
print("\n各簇的平均特征值：")
print(cluster_summary)