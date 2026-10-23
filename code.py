import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

df = pd.read_csv('Car F and P.csv.csv')
print("Data loaded successfully!")
print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
print("Missing Values:")
print(df.isnull().sum()[df.isnull().sum() > 0])
print("\nDescriptive Statistics:")
print(df.describe().round(2))
print("\nStages 1-3 completed.")

print("="*60)
print("Stage 4: Data Preprocessing")
print("="*60)

df_cluster = df.copy()
df_cluster['Market Category'] = df_cluster['Market Category'].fillna('Unknown')
important_num_cols = ['Engine HP', 'Engine Cylinders', 'Number of Doors']
df_cluster = df_cluster.dropna(subset=important_num_cols)
print(f"Shape after handling missing values: {df_cluster.shape}")

numerical_features = ['Year', 'Engine HP', 'Engine Cylinders', 'highway MPG',
                      'city mpg', 'Popularity', 'MSRP']
categorical_features = ['Make', 'Engine Fuel Type', 'Transmission Type',
                        'Driven_Wheels', 'Vehicle Size', 'Vehicle Style']

print("Numerical features:", numerical_features)
print("Categorical features:", categorical_features)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

X = preprocessor.fit_transform(df_cluster)
print(f"\nFinal shape after encoding and scaling: {X.shape}")
print("\nStage 4 completed.")

print("="*60)
print("Stage 5: Elbow Method (sample of 3000 for speed)")
print("="*60)

elbow_idx = np.random.choice(X.shape[0], size=3000, replace=False)
X_elbow = X[elbow_idx]

inertia = []
K_range = range(2, 15)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_elbow)
    inertia.append(km.inertia_)
    print(f"K = {k} | Inertia = {km.inertia_:.2f}")

plt.figure(figsize=(10, 6))
plt.plot(K_range, inertia, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.grid(True)
plt.savefig('elbow_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nElbow plot saved. Stage 5 completed.")

print("="*60)
print("Stage 6: Final Clustering with K=6")
print("="*60)

K = 6
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
df_cluster['Cluster'] = kmeans.fit_predict(X)
print(f"\nK-Means clustering completed with {K} clusters.")
print("\nNumber of cars in each cluster:")
print(df_cluster['Cluster'].value_counts().sort_index())
print("\nAverage key features per cluster:")
key_features = ['Engine HP', 'MSRP', 'highway MPG', 'city mpg', 'Year']
print(df_cluster.groupby('Cluster')[key_features].mean().round(2))
print("\nStage 6 completed successfully.")

print("="*70)
print("Stage 7: Cluster Analysis and Business Interpretation")
print("="*70)

print("1. Number of cars in each cluster:")
print(df_cluster['Cluster'].value_counts().sort_index())
print("\n" + "="*50)
print("2. Average characteristics of each cluster:")
key_features = ['Year', 'Engine HP', 'Engine Cylinders', 'highway MPG',
                'city mpg', 'MSRP', 'Popularity']
print(df_cluster.groupby('Cluster')[key_features].mean().round(2))
print("\n" + "="*50)
print("3. Top 5 Car Brands in each cluster:")
for i in range(6):
    top_brands = df_cluster[df_cluster['Cluster'] == i]['Make'].value_counts().head(5)
    print(f"\nCluster {i}:")
    print(top_brands)
print("\n" + "="*50)
print("4. Most common Vehicle Style in each cluster:")
for i in range(6):
    style = df_cluster[df_cluster['Cluster'] == i]['Vehicle Style'].value_counts().head(3)
    size = df_cluster[df_cluster['Cluster'] == i]['Vehicle Size'].value_counts().head(2)
    print(f"\nCluster {i} -> Style: {style.index[0]} ({style.iloc[0]}) | Size: {size.index[0]}")
print("\nStage 7 completed.")

print("="*70)
print("Stage 8: Silhouette Score Evaluation")
print("="*70)

sample_idx = np.random.choice(X.shape[0], size=3000, replace=False)
X_sample = X[sample_idx]
labels_sample = df_cluster['Cluster'].values[sample_idx]

overall_score = silhouette_score(X_sample, labels_sample)
print(f"\nOverall Silhouette Score: {overall_score:.4f}")
print("   (Scale: -1 to 1 | Above 0.5 = Good | Above 0.7 = Strong)")

sample_silhouette_values = silhouette_samples(X_sample, labels_sample)
print("\nSilhouette Score per Cluster:")
for i in range(K):
    cluster_vals = sample_silhouette_values[labels_sample == i]
    print(f"  Cluster {i}: {cluster_vals.mean():.4f}  (n={len(cluster_vals)})")

fig, ax = plt.subplots(figsize=(10, 7))
y_lower = 10
colors = plt.cm.tab10(np.linspace(0, 1, K))
for i in range(K):
    cluster_vals = sample_silhouette_values[labels_sample == i]
    cluster_vals.sort()
    size_i = len(cluster_vals)
    y_upper = y_lower + size_i
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_vals,
                     facecolor=colors[i], edgecolor=colors[i], alpha=0.8)
    ax.text(-0.05, y_lower + 0.5 * size_i, str(i), fontsize=11)
    y_lower = y_upper + 10
ax.axvline(x=overall_score, color="red", linestyle="--", linewidth=1.5,
           label=f"Overall Score = {overall_score:.3f}")
ax.set_xlabel("Silhouette Coefficient", fontsize=12)
ax.set_ylabel("Cluster", fontsize=12)
ax.set_title("Silhouette Plot - K-Means (K=6)", fontsize=14, fontweight='bold')
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig('silhouette_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nStage 8 completed.")

print("="*70)
print("Stage 9: PCA Cluster Visualization")
print("="*70)

pca_idx = np.random.choice(X.shape[0], size=3000, replace=False)
X_pca_sample = X[pca_idx]
labels_pca_sample = df_cluster['Cluster'].values[pca_idx]

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_pca_sample)
explained = pca.explained_variance_ratio_ * 100
print(f"Variance explained: PC1 = {explained[0]:.1f}% | PC2 = {explained[1]:.1f}% | Total = {explained.sum():.1f}%")

df_pca = pd.DataFrame({
    'PC1': X_pca[:, 0],
    'PC2': X_pca[:, 1],
    'Cluster': labels_pca_sample
})

cluster_labels = {
    0: "Old Economy",
    1: "Family Cars",
    2: "Sports Cars",
    3: "Luxury/Super",
    4: "Fuel Efficient",
    5: "Mid-Range"
}
df_pca['Label'] = df_pca['Cluster'].map(cluster_labels)

fig, ax = plt.subplots(figsize=(13, 9))
palette = sns.color_palette("tab10", K)
for i in range(K):
    mask = df_pca['Cluster'] == i
    ax.scatter(df_pca.loc[mask, 'PC1'], df_pca.loc[mask, 'PC2'],
               c=[palette[i]], label=f"Cluster {i}: {cluster_labels[i]}",
               alpha=0.5, s=15, edgecolors='none')
for i in range(K):
    mask_i = labels_pca_sample == i
    if mask_i.sum() > 0:
        cx, cy = X_pca[mask_i, 0].mean(), X_pca[mask_i, 1].mean()
        ax.scatter(cx, cy, c='black', marker='X', s=200, zorder=5,
                   label='Cluster Centers' if i == 0 else "")
ax.set_xlabel(f"PC1 ({explained[0]:.1f}% variance)", fontsize=12)
ax.set_ylabel(f"PC2 ({explained[1]:.1f}% variance)", fontsize=12)
ax.set_title("K-Means Clusters Visualized with PCA (K=6)", fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10, markerscale=2)
plt.tight_layout()
plt.savefig('pca_clusters.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nStage 9 completed.")

print("="*80)
print("FINAL CLUSTER INTERPRETATION - PROJECT SUMMARY")
print("="*80)
interpretation = {
    0: "Old and cheap economy cars",
    1: "Average family cars",
    2: "Powerful sports cars",
    3: "Luxury and supercars",
    4: "Fuel-efficient economy cars",
    5: "Mid-range balanced cars"
}

for cluster, desc in interpretation.items():
    size = df_cluster[df_cluster['Cluster'] == cluster].shape[0]
    avg_hp = df_cluster[df_cluster['Cluster'] == cluster]['Engine HP'].mean().round(1)
    avg_price = df_cluster[df_cluster['Cluster'] == cluster]['MSRP'].mean().round(0)
    avg_year = df_cluster[df_cluster['Cluster'] == cluster]['Year'].mean().round(1)
    print(f"Cluster {cluster}: {desc}")
    print(f"   Count: {size} cars | Avg HP: {avg_hp} | Avg Price: ${avg_price:,} | Avg Year: {avg_year}")
    print("-" * 70)

print(f"\nOverall Silhouette Score: {overall_score:.4f}")
print("\nProject completed successfully.")