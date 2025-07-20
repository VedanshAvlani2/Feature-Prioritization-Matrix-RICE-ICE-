import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

# --- Load original dataset ---
df = pd.read_csv("feature_prioritization.csv")

# ------------------------
# 1. RICE and ICE
# ------------------------
df["RICE"] = (df["Reach"] * df["Impact"] * df["Confidence"]) / df["Effort"]
df["ICE"] = (df["Impact"] * df["Confidence"] * df["Ease"]) / 100


# Sort top 20 by RICE
top_rice = df.sort_values(by="RICE", ascending=False).head(20)
top_ice = df.sort_values(by="ICE", ascending=False).head(20)

# ------------------------
# 🔁 2. Sensitivity Analysis (Confidence)
# ------------------------
df["RICE_ConfPlus10"] = (df["Reach"] * df["Impact"] * (df["Confidence"] * 1.10)) / df["Effort"]
df["ICE_ConfMinus10"] = (df["Impact"] * (df["Confidence"] * 0.90) * df["Ease"]) / 100
df["RICE_Delta"] = df["RICE_ConfPlus10"] - df["RICE"]
df["ICE_Delta"] = df["ICE_ConfMinus10"] - df["ICE"]

# ------------------------
# 🧮 3. Weighted RICE/ICE (customizable)
# ------------------------
# You can tweak these weights
w_reach = 1.0
w_impact = 1.0
w_conf = 1.0
w_effort = 1.0
w_ease = 1.0

df["Weighted_RICE"] = (w_reach * df["Reach"] * w_impact * df["Impact"] * w_conf * df["Confidence"]) / (w_effort * df["Effort"])
df["Weighted_ICE"] = (w_impact * df["Impact"] * w_conf * df["Confidence"] * w_ease * df["Ease"]) / 100

# ------------------------
# 🧠 4. Clustering Features by Profile
# ------------------------
cluster_features = df[["Reach", "Impact", "Confidence", "Effort", "Ease", "RICE", "ICE"]]
scaled = StandardScaler().fit_transform(cluster_features)
kmeans = KMeans(n_clusters=4, random_state=42)
df["Cluster"] = kmeans.fit_predict(scaled)

# ------------------------
# 🚨 5. Red Flag Detection
# ------------------------
df["Red_Flag"] = np.where(
    (df["Effort"] > 7) & (df["Impact"] < 2),
    "⚠️ High Effort, Low Impact",
    np.where((df["Confidence"] < 60) & (df["Reach"] > 5000), "⚠️ Low Confidence, High Reach", "")
)

# ------------------------
# Visualizations
# ------------------------

# Plot RICE scores
plt.figure(figsize=(10, 6))
sns.barplot(y=top_rice["Feature"], x=top_rice["RICE"], palette="viridis")
plt.title("Top 20 Features by RICE Score")
plt.xlabel("RICE Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# Plot ICE scores
plt.figure(figsize=(10, 6))
sns.barplot(y=top_ice["Feature"], x=top_ice["ICE"], palette="coolwarm")
plt.title("Top 20 Features by ICE Score")
plt.xlabel("ICE Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df[["Reach", "Impact", "Confidence", "Effort", "Ease", "RICE", "ICE"]].corr(), annot=True, cmap="YlGnBu")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

# Sensitivity Scatter
sns.scatterplot(data=df.sample(1000), x="RICE", y="RICE_Delta", hue="Red_Flag")
plt.title("Sensitivity of RICE to +10% Confidence")
plt.show()

# Clusters
sns.pairplot(df.sample(1000), vars=["RICE", "ICE", "Reach", "Impact"], hue="Cluster")
plt.suptitle("KMeans Clustering of Feature Profiles", y=1.02)
plt.show()

# ------------------------
# Save updated dataset
# ------------------------
df.to_csv("feature_prioritization_enhanced.csv", index=False)

# Basic Output
print("✅ Enhanced scoring, clustering, red flags, and sensitivity added.")
