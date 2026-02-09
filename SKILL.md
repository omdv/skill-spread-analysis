---
name: options-spread-analysis
description: Analyzes market conditions for selling option spreads (put or call). Use when the user asks whether to open a bull put spread or bear call spread, wants a market check before trading, or asks which direction looks better for premium selling.
---

# Options Spread Analysis

Analyzes SPX market conditions for both put and call spread selling, showing side-by-side comparison to help you pick the better entry.

## Usage

```bash
# Live data (requires FMP_API_KEY)
uv run scripts/analyze.py

# With plot
uv run scripts/analyze.py --plot

# Save plot
uv run scripts/analyze.py --save=analysis.png

# Synthetic data (no API key)
uv run scripts/analyze.py --synthetic
uv run scripts/analyze.py --synthetic --scenario=rally
uv run scripts/analyze.py --synthetic --scenario=dip
```

**Requirements:** `FMP_API_KEY` environment variable. Get free key at https://financialmodelingprep.com/developer/docs/

---

## Trading Profile (Embedded)

**Last Updated:** February 2026

### Primary Strategy
**SPX Credit Spreads** - Theta collection via far OTM credit spreads

### Position Parameters

| Parameter | Put Spread | Call Spread |
|-----------|------------|-------------|
| Target Delta | 0.06 (94% POP) | 0.04 (96% POP) |
| Preferred DTE | 28 days | 28 days |
| DTE Range | 14-35 days | 14-35 days |
| Position Type | Defined-risk | Defined-risk |

### Risk Management Rules
- **Profit Taking:** Close at 50-75% of max profit
- **VIX Exit Trigger:** Exit positions if VIX exceeds 18-20
- **Trend Monitoring:** Watch SPX technical levels
- **Risk Priority:** Safety over maximum premium collection

### Ideal Market Conditions
- VIX: Below 18 (preferably 12-16 range)
- SPX: Stable or trending in your favor
- IV Percentile: 30+ (acceptable down to 10 with adjustments)
- No major catalysts in next 1-2 weeks

---

## Output Signals

| Signal | Meaning |
|--------|---------|
| `GREEN_LIGHT` | Good conditions for this spread type |
| `GREEN_LIGHT (HIGH CONVICTION)` | Multiple boosters confirm setup |
| `WAIT` | Regime OK but entry not ideal |
| `RED_FLAG` | Do not trade - unfavorable regime |

---

## Analysis Logic

### Regime Filters (Safety - Both Directions)

These MUST pass before considering any spread:

| Filter | Pass | Fail |
|--------|------|------|
| Hurst Exponent | H < 0.5 (mean-reverting) | H > 0.5 (trending) |
| Variance Ratio | VR ≤ 1.0 (no momentum) | VR > 1.0 (momentum) |

**If regime fails → RED_FLAG for both directions**

### Put Spread Entry Conditions

| Condition | Target | Description |
|-----------|--------|-------------|
| Z-Score | < -1.0 | Dip from moving average |
| VIX Direction | Rising or elevated | Rich put premiums |

**Conviction Boosters:**
- Deep dip (Z < -2.0)
- VIX spike (VIX Z-score > 2.0)
- Stationarity confirmed (ADF + KPSS pass)
- Fast reversion (half-life < 10 days)

### Call Spread Entry Conditions

| Condition | Target | Description |
|-----------|--------|-------------|
| Z-Score | > +1.0 | Rally above moving average |
| VIX Direction | Falling | Confirms rally, premiums fair |

**Conviction Boosters:**
- Strong rally (Z > +2.0)
- VIX crushed (VIX Z-score < -1.0)
- Stationarity confirmed
- Fast reversion (half-life < 10 days)

---

## Risk Assessment

When evaluating spreads, consider:

### Market Risks
- Upcoming economic events (Fed meetings, CPI, jobs data)
- Earnings season impact
- Geopolitical events
- Technical levels nearby (support/resistance)

### Volatility Risks
- VIX term structure (contango vs backwardation)
- Recent volatility spikes or compression
- Volatility expansion risk if IV very low

### Position-Specific Risks
- Cushion from current price to short strike
- What needs to happen for trade to lose
- Max loss amount (absolute dollars)

---

## Special Considerations

### Low IV Environment (IV Percentile < 20)
- Smaller credits are expected and acceptable
- Prefer shorter DTE (14-21 over 28)
- Tighter spreads may be better (reduce max risk)
- Be extra disciplined on profit taking
- Watch for volatility expansion carefully

### High IV Environment (VIX > 20)
- Generally avoid NEW short premium positions
- Consider waiting for volatility to normalize
- If already in positions, manage actively
- Focus on trade management rather than new entries

### Transition Periods (VIX 16-18)
- Elevated caution
- Prefer very conservative strikes (0.04-0.05 delta for puts)
- Shorter DTE (14-21 days)
- Smaller position sizes
- Have exit plan ready

---

## Key Principles

1. **Safety First:** Prioritize safety over maximum premium
2. **Discipline:** Don't force trades when conditions don't meet parameters
3. **Consistency:** Stick to target deltas and DTE framework
4. **Risk Management:** VIX above 18 is a hard stop for new short premium
5. **Profit Taking:** Don't get greedy - close at 50-75% of max profit
6. **Market Respect:** When in doubt, stay out - capital preservation matters

---

## Workflow

1. Run `uv run scripts/analyze.py`
2. Check regime filters (Hurst, VR) - if RED_FLAG, stop here
3. Compare put vs call signals side by side
4. Pick the side with better conditions (or wait if both weak)
5. Use specific strike/expiration analysis for final trade decision
6. Set profit target and exit rules before entering

---

## Report Saving

After completing analysis, save a report:

**Location:** `reports/analysis/YYYY-MM-DD.md`

The script outputs structured data that can be saved for tracking.

---

## Limitations & Disclaimers

- Uses EOD data - intraday conditions may differ
- Statistical signals are probabilistic, not guarantees
- VIX direction based on recent trend, can reverse
- Markets can change rapidly; verify before execution
- This is systematic analysis, NOT financial advice
- All options trading involves substantial risk of loss
- Past performance does not guarantee future results

---

## Integration Notes

**Data Requirements:**
- SPX price data (150 days historical + current)
- VIX price data (150 days historical + current)
- FMP API key for live data

**When to re-run:**
- Daily before market open
- After significant intraday moves
- Before placing any new spread order
