import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

def apply_hierarchical_clustering(X_pca):
    # 📌 1. Apply hierarchical clustering (3 clusters)
    hc = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
    y_pred_hc = hc.fit_predict(X_pca)

    # 📌 2. Visualize clustering results
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_pred_hc, palette="coolwarm")
    plt.title("Hierarchical Clustering Result")
    plt.xlabel("PC1")  # Principal Component 1
    plt.ylabel("PC2")  # Principal Component 2
    plt.legend(title="Cluster")
    plt.show()

    # 📌 3. Draw dendrogram
    plt.figure(figsize=(10, 5))
    linkage_matrix = linkage(X_pca, method='ward')
    dendrogram(linkage_matrix)
    plt.title("Dendrogram - Hierarchical Clustering")
    plt.xlabel("Samples")
    plt.ylabel("Distance")
    plt.show()
