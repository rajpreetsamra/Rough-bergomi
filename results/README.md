## Skew Term Structure

- **Objective**: Analyze how implied volatility skew evolves across time-to-expiry buckets.
- **Method**: Linear regression on IV vs log-moneyness for puts, aggregated over several dates.
- **Result**: Short-term options exhibit high skew variability; long-term smiles are flatter and more stable.
- **File**: `mean_skew_term_structure.jpg`


## rBergomi Model vs Market IV – Smile Fit Results (2023-02-01)

This analysis compares the **market implied volatility (IV)** smile with model-implied IVs generated from the **rough Bergomi model**. The figures below display the smile fits and quantitative error metrics across two different maturity buckets on February 1, 2023.

---

### Short-Term Bucket (5–15D)

- **Target DTE:** 6 days  
- **Mean IV Diff:** 0.0146  
- **Mean Price Diff:** 1.2551
  

### Medium-Term Bucket (30–45D)

- **Target DTE:** 37 days  
- **Mean IV Diff:** 0.0507  
- **Mean Price Diff:** 13.4992

### Interpretation

- The short-term smile shows strong alignment between market and model IVs, with small pricing and volatility errors.
- The medium-term smile reveals notable deviations, especially in the wings, suggesting potential underfitting of skew or missing dynamics in the model (e.g., smile convexity).
- The high price error in the medium-term bucket underscores the need for better calibration or more flexible volatility-of-volatility structures.
