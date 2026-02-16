---
name: options-spread-analysis
description: Analyzes market conditions for selling option spreads (put or call). Use when the user asks whether to open a bull put spread or bear call spread, wants a market check before trading, or asks which direction looks better for premium selling.
---

# Options Spread Analysis

Run `uv run ./scripts/analyze.py` and add a 2-3 sentence **Summary** interpreting the results.

## Signals

| Signal | Meaning |
|--------|---------|
| `GREEN_LIGHT` | Good entry conditions |
| `GREEN_LIGHT (HIGH CONVICTION)` | Strong setup with multiple confirmations |
| `WAIT` | Regime OK but no entry signal yet |
| `RED_FLAG` | Don't trade — unfavorable regime (trending/momentum market) |

For detailed trading parameters and analysis logic, see [REFERENCE.md](REFERENCE.md).
