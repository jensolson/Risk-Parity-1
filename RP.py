"""
Jens Olson
jens.olson@gmail.com
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
plt.style.use('ggplot')
sns.set_style('darkgrid')

import scipy.stats as stats
import scipy.cluster.hierarchy as sch
from sklearn.decomposition import PCA
from sklearn.covariance import shrunk_covariance, ledoit_wolf, OAS, EmpiricalCovariance

"""
Global minimum variance portfolio
Pages 7-8 of 
https://faculty.washington.edu/ezivot/econ424/portfolioTheoryMatrix.pdf
"""
def min_var(returns_array, MV_lambda):
    """
    Parameters
    -----------
    returns_array: numpy array of historical returns
    MV_lambda: constant in interval [0, 1)
    
    Returns
    -----------
    Minimum variance portfolio weights
    """
    n_ticks = returns_array.shape[1]
    A = np.zeros([n_ticks+1, n_ticks+1])
    A[:-1, :-1] = 2*np.cov(returns_array.T)
    A[-1, 0:-1] = 1
    A[0:-1, -1] = 1
    z0 = np.zeros([n_ticks+1, 1])
    z0[-1] = MV_lambda
    b = np.zeros([n_ticks+1, 1])
    b[-1] = 1
    z = np.linalg.pinv(A) @ b
    return z[:-1]

"""
Newton method ERC weight calc
per 
http://www.top1000funds.com/wp-content/uploads/2012/08/Efficient-algorithms-for-computing-risk-parity-portfolio-weights.pdf
"""
class NewtonERC():
    """
    Calculates risk parity equal risk contribution weights using
    Newton's method solver. 
    """
    def __init__(self, cov_est=None, cov_window=126, corr_window=504, 
                 lw_shrink=None, newt_lambda0=.5, tol=1e-7):
        """
        Initializes attributes:
        cov_est: Method of robust covariance estimation. Enter 'empirical',
                 'oas' or None
        cov_window: integer, # of days to look back to calculate variance
        corr_window: integer, # of days to look back to calculate correlation
        lw_shrink: float between 0 and 1; in lieu of calculating ledoit wolf
                   at each iteration; used only if cov_est == None
        newt_lambda: float between 0 and 1; lambda regularizaiton constant
        tol: float, error tolerance
        """
        self.cov_est = cov_est
        self.cov_window = cov_window
        self.corr_window = corr_window
        self.lw_shrink = lw_shrink
        self.newt_lambda0 = newt_lambda0
        self.tol = tol
        
    def _get_omega(self, returns):
        """
        Get robust covariance matrix for use in Newton solver.
        
        Parameters
        ----------
        returns: numpy array of return data
        
        Returns
        ----------
        omega: array of shape nxn where n is equal to the number of
               securities invovled
        """
        corr_returns = returns[-self.corr_window:, :]
        cov_returns = returns[-self.cov_window:, :]

        if self.cov_est == 'oas':
            omega = OAS().fit(corr_returns).covariance_*10**4
        elif self.cov_est == 'empirical':
            omega = EmpiricalCovariance().fit(corr_returns).covariance_*10**4
        else:
            corr = np.corrcoef(corr_returns, rowvar=False)
            cov_diag = np.diag(np.sqrt(np.var(cov_returns, axis=0)))
            omega = cov_diag @ corr @ cov_diag
            if self.lw_shrink is None:
                lw = ledoit_wolf(corr_returns)[1]
                omega = shrunk_covariance(omega, shrinkage=lw)*10**4
            else:
                omega = shrunk_covariance(omega, shrinkage=self.lw_shrink)*10**4
        return omega
        
    def _get_F(self, omega, y):
        """
        Writes risk parity problem as a system of nonlinear equations
        
        Parameters
        ---------
        omega: robust covariance matrix of shape nxn, where n is equal
               to the number of securities involved
        y: prior guess of weights plus regularizing lambda term
        
        Returns
        ---------
        F: System of nonlinear equations to be used in Newton method solver
        """        
        x = y[:-1]
        newt_lambda = y[-1]
        F = np.zeros([len(x)+1, 1])
        F[:-1] = omega @ x - newt_lambda/x
        F[-1] = x.sum()-1
        return F
        
    def _get_J(self, omega, y):
        """
        Creates Jacobian matrix of F(y) evaluated at point c
        
        Parameters
        ---------
        omega: robust covariance matrix of shape nxn, where n is equal
               to the number of securities involved
        y: prior guess of weights plus regularizing lambda term
        
        Returns
        ---------
        J: Jacobian matrix of F(y)
        """
        x = y[:-1]
        newt_lambda = y[-1]
        J = np.zeros([len(x)+1, len(x)+1])
        J[:-1, :-1] = omega + newt_lambda*np.diagflat(1/(x**2))
        J[:-1, -1] = -1/x.ravel()
        J[-1, :-1] = 1
        return J
        
    def get_weights(self, returns):
        """
        Solves risk parity problem using Newton's method
        
        Parameters
        ---------
        returns: numpy array of return data
        
        Returns
        ---------
        y: Newton method solution of risk parity weights
        """
        omega = self._get_omega(returns)
        x0 = np.ones([returns.shape[1], 1])/returns.shape[1]
        y0 = np.append(x0, self.newt_lambda0).reshape(-1, 1)
        y = y0 - (np.linalg.pinv(self._get_J(omega, y0)) @ self._get_F(omega, y0))
        error = np.linalg.norm(y-y0, ord=2)
        while error > self.tol:
            y_last = y
            y = y_last - (np.linalg.pinv(self._get_J(omega, y_last)) @ self._get_F(omega, y_last))
            error = np.linalg.norm(y-y_last, ord=2)
        return y[:-1]
        

"""
Calculates risk parity weights optimizing for higher moments.
Optimizes portfolio based on variance, coskewness and cokurtosis. 

Source links:
http://www.quantatrisk.com/2013/01/20/coskewness-and-cokurtosis/
http://www.bfjlaward.com/pdf/26084/024-036_Baitinger_JPM.pdf
http://past.rinfinance.com/agenda/2017/talk/BernhardPfaff.pdf
"""
def get_M2(returns):
    """
    Inputs:
    returns: pandas dataframe of returns data
    
    Returns:
    M2, an N x N covariance matrix
    """
    rets_arr = np.array(returns)
    M2 = np.cov(rets_arr.T)
    return M2
    
def get_M3(returns):
    """
    Inputs:
    returns: pandas dataframe of returns data
    
    Returns:
    M3, an N x N^2 reshaped array of coskewness where N = number of 
    securities in the pandas dataframe
    """
    rets_arr = np.array(returns)
    (rows, cols) = rets_arr.shape
    rets_mu = rets_arr.mean(axis=0).reshape(1, -1)
    rets_centered = rets_arr - rets_mu
    
    rets_skew = np.zeros(([cols]*3))
    for i in range(cols):
        for j in range(cols):
            for k in range(cols):
                rets_skew[i, j, k] = np.sum(rets_centered[:, i]*rets_centered[:, j]*rets_centered[:, k])
    
    M3 = np.concatenate([rets_skew[0], rets_skew[1]], axis=1)
    for p in range(2, cols):
        M3 = np.concatenate([M3, rets_skew[p]], axis=1)
    
    M3 /= (rows-1)
    return M3

def get_M4(returns):
    """
    Inputs:
    returns: pandas dataframe of returns data
    
    Returns:
    M4, an N x N^3 reshaped array of cokurtosis where N = number of 
    securities in the pandas dataframe
    """
    rets_arr = np.array(returns)
    (rows, cols) = rets_arr.shape
    rets_mu = rets_arr.mean(axis=0).reshape(1, -1)
    rets_centered = rets_arr - rets_mu

    rets_skew = np.zeros(([cols]*4))
    for i in range(cols):
        for j in range(cols):
            for k in range(cols):
                for l in range(cols):
                    rets_skew[i, j, k, l] = np.sum(rets_centered[:, i]*rets_centered[:, j]\
                                                   *rets_centered[:, k]*rets_centered[:, l])

    M4 = np.concatenate([rets_skew[0, 0], rets_skew[0, 1]], axis=1)
    for p in range(2, cols):
        M4 = np.concatenate([M4, rets_skew[0, p]], axis=1)

    for q in range(1, cols):
        for r in range(0, cols):
            M4 = np.concatenate([M4, rets_skew[q, r]], axis=1)

    M4 /= (rows-1)
    return M4
    
def get_ARC2(M2, w):
    MRC2 = 2*(M2 @ w)
    ARC2 = MRC2 * w
    return ARC2

def get_ARC3(M3, w):
    MRC3 = 3*(M3 @ np.kron(w, w))
    ARC3 = MRC3 * w
    return ARC3

def get_ARC4(M4, w):
    MRC4 = 4*(M4 @ np.kron(np.kron(w, w), w))
    ARC4 = MRC4 * w
    return ARC4

def high_moment_F(w, returns, lambd2, lambd3, lambd4):
    M2 = get_M2(returns)
    M3 = get_M3(returns)
    M4 = get_M4(returns)
    ARC2 = get_ARC2(M2, w)
    ARC3 = get_ARC3(M3, w)
    ARC4 = get_ARC4(M4, w)
    F = lambd2*np.var(ARC2) + lambd3*np.var(ARC3) + lambd4*np.var(ARC4)
    return F*10**8   # numerical instability with smaller numbers
    
"""
Diversified Risk Parity
Based on Harald Lohre's paper:
http://www.northinfo.com/documents/515.pdf
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1974446

Weights are determined so that each eigenvector of the covariance matrix
contributes equally to portfolio risk.
"""
def getDiversifiedWeights(w, sigma):
    """ Parameters:
        w: numpy array of initial weight guesses
        sigma: numpy array of covariance matrix of historical returns
            
        Returns:
        N_Ent: Number of independent orthogonal bets given portfolio weights
               (to be minimized using scipy's solver)
    """
    eigvals, eigvecs = np.linalg.eig(sigma)
    eigvals = eigvals.reshape(-1, 1)
    
    w_tilde = eigvecs.T @ w
    v = w_tilde**2 * eigvals
    p = v/v.sum()
    N_Ent = np.exp(-np.sum(p*np.log(p)))
    return -N_Ent
    
"""
Inspired by DRP above, weights determined so that each principal 
component of return array carries equal loading
"""
def getPCAWeights(w0, returns_array):
    """ Parameters:
        w0: numpy array of initial weight guesses
        returns_array: numpy array of historical returns
        
        Returns:
        sqd_diffs: Object of scipy minimization exercise
    """
    wtd_returns = w0.T * returns_array
    PCA_returns = PCA().fit_transform(wtd_returns)
    std_PCA = PCA_returns.std(axis=0)
    pct_PCA = std_PCA/std_PCA.sum()
    pct_diff = pct_PCA - 1/len(w0)
    sqd_diffs = 100*np.sum(pct_diff**2)
    return sqd_diffs
    
