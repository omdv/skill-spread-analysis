# SPX Mean Reversion Analysis

Statistical mean reversion testing for SPX put-selling decisions.

## Install

```bash
uv sync
```

## Usage

```bash
# With live data (requires FMP API key)
export FMP_API_KEY=your_key
uv run python main.py

# Save plot
uv run python main.py --save=analysis.png

# Display plot
uv run python main.py --plot

# Synthetic data only (no API key needed)
uv run python main.py --synthetic
```

Get a free API key at https://financialmodelingprep.com/developer/docs/

## Signal Logic

| Signal | Meaning |
|--------|---------|
| `RED_FLAG` | Don't trade - trending/momentum regime |
| `WAIT` | Regime OK but no dip - wait for pullback |
| `GREEN_LIGHT` | Dip detected in mean-reverting regime |

## Key Metrics

- **Hurst Exponent**: < 0.5 = mean-reverting, > 0.5 = trending
- **Z-Score**: Current stretch normalized by volatility
- **Half-Life**: Days to recover halfway to mean (sets DTE)
- **VIX Z-Score**: Elevated = rich premiums (conviction booster)

## Library Usage

```python
from main import analyze_mean_reversion, fetch_spx_and_vix, get_current_price

# Fetch data
spx_df, vix_series = fetch_spx_and_vix(days=150)
current_spx = get_current_price("^GSPC")
current_vix = get_current_price("^VIX")

# Analyze
result = analyze_mean_reversion(
    spx_df,
    current_spx,
    vix_prices=vix_series,
    current_vix=current_vix,
)

print(result.signal)
print(f"Recommended DTE: {result.half_life.recommended_min_dte}-{result.half_life.recommended_max_dte}")
```
