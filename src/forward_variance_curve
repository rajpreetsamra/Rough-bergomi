#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import regex as re
from scipy.stats import norm
from scipy.optimize import brentq
import datetime


# In[13]:


def fbm_cholesky(T, N, H):
    dt = T / N
    times = np.linspace(0, T, N + 1)
    def gamma(s, t):
        return 0.5 * (s**(2 * H) + t**(2 * H) - abs(s - t)**(2 * H))
    cov = np.fromfunction(lambda i, j: gamma(times[i], times[j]), (N + 1, N + 1), dtype=int)
    cov += 1e-10 * np.eye(N + 1)
    L = np.linalg.cholesky(cov)
    return L


# In[15]:


def kernel_weight(H, steps):
    N = len(steps) - 1               
    dt = steps[1] - steps[0]         
    K = np.zeros((N, N))           
    for n in range(N):              
        for k in range(n + 1):      
            delta = (n + 1 - k) * dt
            K[n, k] = delta ** (H - 0.5)   
    return K


# In[17]:


def fbm_increments(path):
    increment=[]
    for i in range(len(path)-1):
        increment.append(path[i+1]-path[i])
    return np.array(increment)
    
    


# In[20]:


def generate_forward_variance_curves(df_options, H, eta, epsilon_0, M, N_per_year=1512, plot=True):
    print(H,eta,epsilon_0)
    unique_dtes = sorted(df_options["DTE"].unique())
    T_max = max(unique_dtes) / 252
    N = int(np.ceil(T_max * N_per_year))
    dt = T_max / N
    steps = np.linspace(0, T_max, N + 1)
    kernel_full = kernel_weight(H, steps)
    L_full = fbm_cholesky(T_max, N, H)
    W_t = np.random.randn(N + 1, M)
    fbm_paths = L_full @ W_t
    increments_full = fbm_increments(fbm_paths)
    variance_curves_by_dte = {}

    for dte in unique_dtes:
        T = dte / 252
        N_T = int(T * N_per_year)
        kernel_T = kernel_full[:N_T, :N_T]
        increments_T = increments_full[:N_T, :]
        integral = kernel_T @ increments_T
        integral = np.clip(integral, -10, 10)
        fwd_var = epsilon_0 * np.exp(eta * integral)
        fwd_var = np.clip(fwd_var, 1e-6, 5.0)
        variance_curves_by_dte[dte] = fwd_var
        if plot:
            time_grid = np.linspace(0, T, N_T)
            avg_path = np.mean(fwd_var, axis=1)
            plt.figure(figsize=(8, 4))
            plt.plot(time_grid, avg_path, label=f"DTE {dte} | T={T:.2f} years")
            plt.title(f"Forward Variance Curve – DTE {dte}")
            plt.xlabel("Time (Years)")
            plt.ylabel("Forward Variance")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()
    print("\n==== Summary ====")
    print(f"Global mean ξ: {np.mean([v.mean() for v in variance_curves_by_dte.values()]):.4f}")
    print(f"Global std ξ:  {np.std([v.mean() for v in variance_curves_by_dte.values()]):.4f}")


    return variance_curves_by_dte
