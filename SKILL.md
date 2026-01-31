---
name: mean-reversion-test
description: Analyze SPX mean reversion for put-selling decisions. Use when the user wants to check market conditions, run mean reversion analysis, generate trading signals (GREEN_LIGHT/WAIT/RED_FLAG), or analyze SPX/VIX relationship for options trading.
compatibility: Requires uv, python 3.14, and optionally FMP_API_KEY for live data
metadata:
  author: om
  version: "1.0"
---

# SPX Mean Reversion Analysis

## Instructions

Follow these steps to run the analysis and interpret results for the user:

### Step 1: Run the Analysis

Execute the appropriate command based on context:

```bash
# Default (uses live data if FMP_API_KEY is set, otherwise synthetic)
uv run python main.py

# With plot saved to file
uv run python main.py --save=analysis.png
```

### Step 2: Interpret the Signal

After running, explain the signal to the user in plain language:

| Signal | What to Tell the User |
|--------|----------------------|
| **GREEN_LIGHT** | "Conditions look favorable for selling puts. The market is in a mean-reverting regime and we have a dip." |
| **GREEN_LIGHT (HIGH CONVICTION)** | "This is a strong setup. Multiple factors align: [list the conviction factors from output]." |
| **GREEN_LIGHT (A+ SETUP)** | "This is an exceptional setup with 3+ confirming factors. High-conviction opportunity." |
| **WAIT** | "The market is mean-reverting (good), but there's no significant dip yet. Wait for a pullback." |
| **RED_FLAG** | "Do not sell puts right now. The market is trending/in momentum mode - mean reversion strategies will fail." |

### Step 3: Highlight Key Numbers

Extract and explain these metrics from the output:

1. **Z-Score**: How stretched is the market?
   - Below -2: "Deep dip - rubber band is very stretched"
   - -2 to -1: "Moderate dip"
   - Above -1: "No significant dip"

2. **Hurst Exponent**: Is mean reversion working?
   - Below 0.45: "Strongly mean-reverting - good for the strategy"
   - 0.45-0.55: "Neutral/random walk"
   - Above 0.55: "Trending - dangerous for put selling"

3. **Half-Life & Recommended DTE**: How long until reversion?
   - Tell user: "Based on half-life of X days, target DTE of Y-Z days for your puts"

4. **VIX** (if shown): Are premiums rich?
   - Z-Score > 2: "VIX is elevated - premiums are rich, good time to sell"
   - Otherwise: "VIX is normal - standard premiums"

### Step 4: Give Actionable Summary

End with a clear recommendation, e.g.:
- "Bottom line: [GREEN/WAIT/RED]. [One sentence on what to do]. If entering, target X-Y DTE."

## Environment Note

If the script shows "FMP_API_KEY not set", inform the user:
- Analysis ran on synthetic demo data
- For live SPX/VIX data: `export FMP_API_KEY=your_key`
- Free API key at https://financialmodelingprep.com/developer/docs/
