# rBergomi Volatility Smile Calibration

This repository implements a volatility surface calibration pipeline using the rough Bergomi (rBergomi) stochastic volatility model. It includes market data preprocessing, forward variance curve simulation, implied volatility smile fitting, and visual analysis of smile fits across maturities and dates.

## 📂 Project Structure

```
.
├── main.py                   # Entry script for end-to-end execution
├── calibration.py           # Parameter setup 
├── market_preprocessing.py  # Preprocessing raw options data
├── pricing_options.py       # Pricing using the rBergomi model
├── forward_variance_curve.py# Simulation of forward variance and Brownian paths
├── results/                 # Output folder for IV smile plots and metrics
```

## 📊 Results

The calibration framework compares **market IV** and **rBergomi-implied IV** across moneyness buckets and maturities.

### Example Output (2023-02-01):

#### Short-term (5–15D), DTE ≈ 6
- Mean IV Diff: `0.0146`
- Mean Price Diff: `1.2551`

#### Medium-term (30–45D), DTE ≈ 37
- Mean IV Diff: `0.0507`
- Mean Price Diff: `13.4992`

### Visualizations

Plots show implied volatility smiles across log-moneyness buckets for different DTE bands, overlaying market and model IVs.
