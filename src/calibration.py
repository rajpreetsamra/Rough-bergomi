#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import pandas_datareader.data as web
import os
import regex as re
from scipy.stats import norm
from scipy.optimize import brentq


# In[ ]:


from scipy.optimize import minimize

big_errors = []
optimal_params = {}

for i in range(len(dates)):
    print(f"[{i+1}/{len(dates)}] Date: {dates[i]}")    
    sampled_options = sampled_options_all_days[sampled_options_all_days["date"] == dates[i]]
    options_df = sampled_options.copy()
    df_short = options_df[options_df["DTE"] <= 5]
    df_atm = df_short[np.abs(df_short["strike_price"] - df_short["SP500"]) / df_short["SP500"] < 0.01]

    if df_atm.empty or df_atm["impl_volatility"].isna().all():
        print("No ATM options found for DTE ≤ 5 with valid IV")
        epsilon_0 = 0.04 
    else:
        sigma_atm = df_atm["impl_volatility"].mean()
        epsilon_0 = sigma_atm ** 2
        print(f"ATM IV mean = {sigma_atm:.4f}, ε₀ = {epsilon_0:.6f}")
    
    def calibration_loss_filtered(theta):
        H, eta, rho = theta
        fwd = generate_forward_variance_curves(options_df, H, eta, epsilon_0, M=3000, plot=False)
        bm1 = get_shared_BM_from_dataframe(options_df, M=3000)
        bm2 = get_shared_BM_from_dataframe(options_df, M=3000)

        options_sample = pricing_options_dataframe(
            options_df=options_df,
            fwd_var_dict=fwd,
            shared_W_dict=bm1,
            shared_W_perp_dict=bm2,
            r=0.037,
            rho=rho,
            option_type='put'
        )

        valid_buckets = ["ITM", "ATM-ish", "OTM", "Far OTM"]
        filtered_df = options_sample[options_sample["moneyness_bucket"].isin(valid_buckets)]

        if filtered_df.empty:
            return 1e3  

        filtered_df["abs_diff_iv"] = (filtered_df["model_iv"] - filtered_df["market_iv"]).abs()
        mean_abs_diff = filtered_df["abs_diff_iv"].mean()
        return mean_abs_diff


    def loss_with_print(theta):
        loss = calibration_loss_filtered(theta)
        print(f"H={theta[0]:.4f}, eta={theta[1]:.4f}, rho={theta[2]:.4f}, IV Abs Diff={loss:.6f}")
        return loss


    initial_theta = [0.4, 0.02, -0.7]
    bounds = [
        (0.01, 0.49),   # H
        (0.01, 5.0),    # eta
        (-0.999, 0.0),  # rho
    ]

    result = minimize(
        loss_with_print,
        x0=initial_theta,
        method='L-BFGS-B',
        bounds=bounds,
        options={'disp': True, 'maxiter': 50}
    )

    if result.success:
        H_opt, eta_opt, rho_opt = result.x
        print(f"Calibrated params: H={H_opt:.4f}, eta={eta_opt:.4f}, rho={rho_opt:.4f}")
        optimal_params[dates[i]] = result.x
    else:
        print(f"Calibration failed for {dates[i]}")
        big_errors.append(i)


# In[ ]:


print("\n Final Calibration Result ")
print(f"Optimal H   = {result.x[0]:.6f}")
print(f"Optimal eta = {result.x[1]:.6f}")
print(f"Optimal rho = {result.x[2]:.6f}")
print(f"Final Loss  = {result.fun:.6f}")
