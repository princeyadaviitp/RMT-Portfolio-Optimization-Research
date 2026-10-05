import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rmt_engine import RMTFilter

def plot_eigenvalue_spectrum(prices, title="RMT: Signal vs Noise"):
    returns = prices.pct_change().dropna()
    T, N = returns.shape
    q = N / T
    sigma = 1 

    corr = returns.corr()
    eVal, _ = np.linalg.eigh(corr)
    eVal = eVal[::-1] # Sort descending

    rmt = RMTFilter()
    pdf_series = rmt._fit_marchenko_pastur(var=sigma, q=q, pts=1000)

    lambda_max = sigma * (1 + np.sqrt(q))**2

    plt.figure(figsize=(12, 6))

    sns.histplot(eVal, bins=50, kde=False, stat="density", 
                 label="Empirical Eigenvalues", color="skyblue", alpha=0.6)

    plt.plot(pdf_series.index, pdf_series.values, color='red', linewidth=2, 
             label="Marchenko-Pastur PDF (Noise)")

    plt.axvline(x=lambda_max, color='green', linestyle='--', linewidth=2, 
                label=f"Noise Threshold ($\lambda_+ = {lambda_max:.2f}$)")

    plt.text(lambda_max * 1.1, plt.ylim()[1]*0.8, "Signal Region \n(True Correlations)", 
             color='green', fontweight='bold')
    plt.text(lambda_max * 0.5, plt.ylim()[1]*0.5, "Noise Region \n(Random Coincidences)", 
             color='red', ha='center')

    plt.title(title, fontsize=14)
    plt.xlabel("Eigenvalue ($\lambda$)")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    from main import generate_synthetic_data

    print("Generating market data...")
    prices = generate_synthetic_data(n_assets=500, n_obs=1000)
    
    print("Plotting Spectrum...")
    plot_eigenvalue_spectrum(prices)
