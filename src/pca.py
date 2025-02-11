import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

def apply_pca():
    # 📌 1. Load dataset
    df = pd.read_csv("../data/Iris.csv")


    # 📌 2. Remove unnecessary column (if ID exists)
    if 'Id' in df.columns:
        df.drop(columns=['Id'], inplace=True)

    # 📌 3. Select numerical columns
    X = df.iloc[:, :-1].values  # All columns except last one

    # 📌 4. Apply PCA (reduce to 2 dimensions)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # 📌 5. Visualize PCA results
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df.iloc[:, -1], palette="viridis")
    plt.title("PCA - Reduced to 2 Dimensions")
    plt.xlabel("PC1")  # Principal Component 1
    plt.ylabel("PC2")  # Principal Component 2
    plt.legend(title="Flower Species")
    plt.show()

    return X_pca  # Return PCA result for further use
