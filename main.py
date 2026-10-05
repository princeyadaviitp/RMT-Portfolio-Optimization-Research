import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, objective_functions, HRPOpt
from rmt_engine import RMTFilter

N_ASSETS = 150       
N_OBSERVATIONS = 1200
RANDOM_SEED = 42

def generate_synthetic_data(n_assets, n_obs):
    np.random.seed(RANDOM_SEED)

    factors = np.random.normal(size=(n_obs, 10)) 
    loadings = np.random.normal(size=(n_assets, 10))
    
    systematic_returns = np.dot(factors, loadings.T)
    noise = np.random.normal(size=(n_obs, n_assets)) * 2.5 
    
    returns = systematic_returns + noise
    price_paths = (1 + returns/100).cumprod(axis=0)
    return pd.DataFrame(price_paths, columns=[f'Stock_{i}' for i in range(n_assets)])

def optimize_markowitz(cov_matrix, expected_returns):
    ef = EfficientFrontier(expected_returns, cov_matrix)
    ef.add_objective(objective_functions.L2_reg, gamma=0.1)
    weights = ef.max_sharpe()
    return ef.clean_weights()

def optimize_hrp(returns=None, cov_matrix=None):
    if cov_matrix is not None:
        hrp = HRPOpt(cov_matrix=cov_matrix)
    else:
        hrp = HRPOpt(returns=returns)
        
    hrp.optimize()
    return hrp.clean_weights()

def get_oos_performance(weights, returns):
    w_vec = np.array([weights[col] for col in returns.columns])
    port_returns = returns.dot(w_vec)
    sharpe = (port_returns.mean() / port_returns.std()) * np.sqrt(252)
    return sharpe

def main():
    print("--- 1. Generating Complex Market Data ---")
    prices = generate_synthetic_data(N_ASSETS, N_OBSERVATIONS)
    returns = prices.pct_change().dropna()

    train_size = int(N_OBSERVATIONS * 0.7)
    train_data = prices.iloc[:train_size]
    train_returns = returns.iloc[:train_size]
    test_returns = returns.iloc[train_size:]
    
    print(f"Data Shape: {prices.shape}")
    print(f"Noise Level: High (Sigma=2.5)")

    print("\n--- 2. Matrix Engineering ---")

    sample_cov = train_returns.cov()

    rmt = RMTFilter()
    rmt_cov = rmt.fit_transform(train_data)

    mu = train_returns.mean() * 252

    print("\n--- 3. The 4-Way Optimization Battle ---")

    print("1. Running Standard Markowitz...")
    w_std = optimize_markowitz(sample_cov, mu)

    print("2. Running RMT + Markowitz...")
    w_rmt_mpt = optimize_markowitz(rmt_cov, mu)

    print("3. Running Standard HRP...")
    w_hrp = optimize_hrp(returns=train_returns)

    print("4. Running RMT + HRP (God Tier)...")
    w_god_tier = optimize_hrp(cov_matrix=rmt_cov)

    print("\n--- 4. Final Results (Out-of-Sample Sharpe) ---")
    
    sharpe_std = get_oos_performance(w_std, test_returns)
    sharpe_rmt_mpt = get_oos_performance(w_rmt_mpt, test_returns)
    sharpe_hrp = get_oos_performance(w_hrp, test_returns)
    sharpe_god = get_oos_performance(w_god_tier, test_returns)

    results = {
        "Standard Markowitz": sharpe_std,
        "RMT + Markowitz": sharpe_rmt_mpt,
        "Standard HRP": sharpe_hrp,
        "RMT + HRP (God Tier)": sharpe_god
    }

    sorted_results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))

    print("-" * 50)
    print(f"{'Strategy':<25} | {'Sharpe Ratio':<15}")
    print("-" * 50)
    for name, score in sorted_results.items():
        print(f"{name:<25} | {score:.4f}")
    print("-" * 50)

    baseline = results["Standard Markowitz"]
    best_score = max(results.values())
    lift = ((best_score - baseline) / abs(baseline)) * 100
    
    print(f"\n>>> Best Strategy Improvement over Baseline: {lift:.2f}%")

if __name__ == "__main__":
    main()
