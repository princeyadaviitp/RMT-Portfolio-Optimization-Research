import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.neighbors import KernelDensity

class RMTFilter:

    def __init__(self, alpha: float = 0.0):
        self.alpha = alpha

    def _get_pca(self, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        eVal, eVec = np.linalg.eigh(matrix)
        indices = eVal.argsort()[::-1]
        eVal, eVec = eVal[indices], eVec[:, indices]
        return eVal, eVec

    def _fit_marchenko_pastur(self, var: float, q: float, pts: int = 1000) -> pd.Series:
        e_min = var * (1 - np.sqrt(q))**2
        e_max = var * (1 + np.sqrt(q))**2

        e_val = np.linspace(e_min, e_max, pts)

        pdf = (1 / (2 * np.pi * var * q * e_val)) * \
              np.sqrt(np.clip((e_max - e_val) * (e_val - e_min), 0, None))
              
        return pd.Series(pdf, index=e_val)

    def _find_max_eval(self, eVal: np.ndarray, q: float, bWidth: float = 0.01) -> float:
        kde = KernelDensity(kernel='gaussian', bandwidth=bWidth).fit(eVal.reshape(-1, 1))
        x = np.linspace(eVal.min(), eVal.max(), 1000)
        log_dens = kde.score_samples(x.reshape(-1, 1))
        kde_pdf = np.exp(log_dens)

        var = 1
        pdf0 = self._fit_marchenko_pastur(var, q)

        def err_func(x_var):
            pdf = self._fit_marchenko_pastur(x_var[0], q)
            # Align indices for comparison
            return np.sum((pdf - pd.Series(kde_pdf, index=x).reindex(pdf.index).fillna(0))**2)

        opt = minimize(err_func, x0=np.array([0.5]), bounds=((1e-5, 1 - 1e-5),))
        var = opt.x[0]

        return var * (1 + np.sqrt(q))**2

    def _denoise_corr(self, eVal: np.ndarray, eVec: np.ndarray, n_facts: int) -> np.ndarray:
        eVal_ = eVal.copy()
        eVal_[n_facts:] = eVal_[n_facts:].mean()

        corr = np.dot(eVec, eVal_ * eVec.T)

        cov = np.diag(corr)
        corr = np.clip(corr / np.sqrt(np.outer(cov, cov)), -1, 1)
        return corr

    def fit_transform(self, prices: pd.DataFrame) -> pd.DataFrame:
        returns = prices.pct_change().dropna()
        T, N = returns.shape
        q = N / T

        sample_cov = returns.cov()
        sample_corr = returns.corr()

        eVal, eVec = self._get_pca(sample_corr.values)

        max_eval = self._find_max_eval(eVal, q)

        n_facts = eVal[eVal > max_eval].shape[0]
        
        if n_facts < 1:
            n_facts = 1 

        denoised_corr = self._denoise_corr(eVal, eVec, n_facts)

        std_devs = np.sqrt(np.diag(sample_cov))
        denoised_cov = np.outer(std_devs, std_devs) * denoised_corr
        
        return pd.DataFrame(denoised_cov, index=sample_cov.index, columns=sample_cov.columns)
