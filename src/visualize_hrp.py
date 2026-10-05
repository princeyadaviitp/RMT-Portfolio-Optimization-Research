import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from main import generate_synthetic_data

def plot_dendrogram(prices, title="HRP: Asset Clustering"):
    returns = prices.pct_change().dropna()
    corr = returns.corr()

    dist = np.sqrt(2 * (1 - corr))

    linkage = sch.linkage(dist, method='single')

    plt.figure(figsize=(12, 6))
    sch.dendrogram(linkage, labels=returns.columns, leaf_rotation=90)
    
    plt.title(title, fontsize=14)
    plt.ylabel("Euclidean Distance")
    plt.xlabel("Assets")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import numpy as np
    
    print("Generating market data...")
    prices = generate_synthetic_data(n_assets=50, n_obs=1000)
    
    print("Plotting Dendrogram...")
    plot_dendrogram(prices)
