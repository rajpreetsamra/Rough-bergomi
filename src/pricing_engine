#!/usr/bin/env python
# coding: utf-8

# In[1]:


def get_shared_BM_from_dataframe(df, M, N_per_year=1512):
    unique_dtes = sorted(df["DTE"].unique())
    T_max = max(unique_dtes) / 252
    N_max = int(np.ceil(T_max * N_per_year))
    print(f"[BM INFO] Max T = {T_max:.4f} years | N_max = {N_max} steps | M = {M} paths")
    dt = T_max / N_max
    dZ = np.sqrt(dt) * np.random.randn(N_max, M)
    Z_t = np.zeros((N_max + 1, M))
    Z_t[1:, :] = np.cumsum(dZ, axis=0)
    bm_slices = {}
    for dte in unique_dtes:
        T = dte / 252
        N = int(np.ceil(T * N_per_year))
        bm_slices[dte] = Z_t[:N + 1, :]
        print(f"[BM SLICE] DTE = {dte} → N = {N} → Shape: {bm_slices[dte].shape}")
    return bm_slices


# In[2]:


def euler_discretization_multi(S0, forward_variance_curves, Z_t, delta_t):
    N, M = forward_variance_curves.shape  
    asset_paths = np.zeros((N + 1, M))    
    asset_paths[0, :] = S0
    for i in range(N):
        dZ = Z_t[i + 1, :] - Z_t[i, :]                   
        vol = np.sqrt(forward_variance_curves[i, :])     
        drift = -0.5 * forward_variance_curves[i, :] * delta_t
        diffusion = vol * dZ
        asset_paths[i + 1, :] = asset_paths[i, :] * np.exp(drift + diffusion)
    return asset_paths


# In[3]:


def pricing_options_dataframe(options_df, fwd_var_dict, shared_W_dict, shared_W_perp_dict, r=0.037, rho=-0.8, option_type='put'):
    model_prices = []
    model_ivs = []
    market_ivs = []
    for _, row in options_df.iterrows():
        S0 = row["SP500"]
        K = row["strike_price"]
        T = row["DTE"] / 365
        iv_market = row["impl_volatility"]
        DTE = row["DTE"]
        if T < 3 / 365 or pd.isna(iv_market) or DTE not in fwd_var_dict or DTE not in shared_W_dict or DTE not in shared_W_perp_dict:
            model_prices.append(np.nan)
            model_ivs.append(np.nan)
            market_ivs.append(iv_market)
            continue
        fwd_var = fwd_var_dict[DTE]
        W_t = shared_W_dict[DTE]
        W_perp = shared_W_perp_dict[DTE]
        Z_t = rho * W_t + np.sqrt(1 - rho ** 2) * W_perp

        N = fwd_var.shape[0] - 1
        dt = T / N

        S_path = euler_discretization_multi(S0, fwd_var, Z_t, dt)
        terminal_prices = S_path[-1]

        if option_type == 'call':
            payoff = np.maximum(terminal_prices - K, 0)
        else:
            payoff = np.maximum(K - terminal_prices, 0)

        price_today = np.exp(-r * T) * payoff.mean()

        intrinsic = max(0.0, S0 - K) if option_type == 'call' else max(0.0, K - S0)
        if intrinsic == 0.0 and price_today < 1e-4:
            time_value_floor = max(0.5, 0.01 * S0 * np.sqrt(T))  
            price_today = time_value_floor
        price_today = max(price_today, S0 - K - 1e-6) if option_type == 'call' else max(price_today, K - S0 - 1e-6)
        iv_model = implied_volatility(price_today, S0, K, T, r, option_type)

        model_prices.append(price_today)
        model_ivs.append(iv_model)
        market_ivs.append(iv_market)
    options_df = options_df.copy()
    options_df["model_price"] = model_prices
    options_df["model_iv"] = model_ivs
    options_df["market_iv"] = market_ivs

    return options_df
