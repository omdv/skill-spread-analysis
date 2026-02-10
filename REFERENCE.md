# Options Spread Analysis — Reference

Detailed trading parameters and analysis logic. Read this only if the user asks "why" or wants deeper explanation.

---

## Trading Profile

**Strategy:** SPX Credit Spreads — Theta collection via far OTM credit spreads

| Parameter | Put Spread | Call Spread |
|-----------|------------|-------------|
| Target Delta | 0.06 (94% POP) | 0.04 (96% POP) |
| Preferred DTE | 28 days | 28 days |
| DTE Range | 14-35 days | 14-35 days |

**Risk Management:**
- Close at 50-75% of max profit
- Exit if VIX exceeds 18-20
- No new short premium when VIX > 18

---

## Analysis Logic

### Regime Filters

| Filter | Pass | Fail |
|--------|------|------|
| Hurst Exponent | H < 0.5 (mean-reverting) | H > 0.5 (trending) |
| Variance Ratio | VR ≤ 1.0 (no momentum) | VR > 1.0 (momentum) |

If regime fails → RED_FLAG for both directions.

### Put Spread Entry

- Z-Score < -1.0 (dip from MA)
- VIX rising or elevated (rich put premiums)
- Boosters: deep dip (Z < -2), VIX spike, stationarity confirmed, fast reversion

### Call Spread Entry

- Z-Score > +1.0 (rally above MA)
- VIX falling (confirms rally)
- Boosters: strong rally (Z > +2), VIX crushed, stationarity confirmed, fast reversion

---

## VIX Zones

| VIX Level | Action |
|-----------|--------|
| < 16 | Normal conditions |
| 16-18 | Elevated caution, smaller size, shorter DTE |
| > 18 | No new short premium positions |
| > 20 | Avoid entirely, manage existing positions |

---

## Limitations

- Uses EOD data — intraday conditions may differ
- Statistical signals are probabilistic, not guarantees
- This is systematic analysis, NOT financial advice
