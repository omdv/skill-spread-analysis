---
name: mean-reversion-test
description: Analyze SPX mean reversion for put-selling decisions. Use when the user wants to check market conditions, run mean reversion analysis, generate trading signals (GREEN_LIGHT/WAIT/RED_FLAG), or analyze SPX/VIX relationship for options trading.
compatibility: Requires uv, python 3.14, and optionally FMP_API_KEY for live data
metadata:
  author: om
  version: "1.0"
---

# SPX Mean Reversion Analysis

This skill runs statistical mean reversion analysis on SPX to generate trading signals for put-selling strategies.

## Running the Analysis

### Live Data (requires FMP_API_KEY)

```bash
cd /home/om/projects/reversion-test && uv run python main.py
```

### Synthetic Data Demo

```bash
cd /home/om/projects/reversion-test && uv run python main.py --synthetic
```

### With Visualization

```bash
cd /home/om/projects/reversion-test && uv run python main.py --plot
```

### Save Plot to File

```bash
cd /home/om/projects/reversion-test && uv run python main.py --save=analysis.png
```

## Signal Interpretation

| Signal | Action | Meaning |
|--------|--------|---------|
| GREEN_LIGHT | Sell puts | Mean-reverting regime with dip detected |
| GREEN_LIGHT (HIGH CONVICTION) | Sell puts confidently | Multiple confirming factors present |
| GREEN_LIGHT (A+ SETUP) | Best setup | 3+ conviction factors aligned |
| WAIT | Do nothing | Mean-reverting but no dip present |
| RED_FLAG | Do not trade | Trending/momentum regime - mean reversion will fail |

## Key Metrics

- **Z-Score**: Stretch from MA normalized by volatility. < -2 is deep dip.
- **Hurst Exponent**: < 0.5 = mean-reverting (safe), > 0.5 = trending (danger)
- **Half-Life**: Days for stretch to revert halfway. Use for DTE selection.
- **Recommended DTE**: 1x to 2x half-life in days
- **VIX Z-Score**: > 2 means elevated premiums (conviction booster)

## Environment Setup

For live SPX/VIX data, set `FMP_API_KEY`:

```bash
export FMP_API_KEY=your_key_here
```

Get a free key at https://financialmodelingprep.com/developer/docs/

Without the API key, the script uses synthetic data that demonstrates the analysis.
