from pca import apply_pca
from hierarchical_clustering import apply_hierarchical_clustering

if __name__ == "__main__":
    print("Running PCA...")
    X_pca = apply_pca()

    print("Running Hierarchical Clustering...")
    apply_hierarchical_clustering(X_pca)
