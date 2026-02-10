---
name: options-spread-analysis
description: Analyzes market conditions for selling option spreads (put or call). Use when the user asks whether to open a bull put spread or bear call spread, wants a market check before trading, or asks which direction looks better for premium selling.
---

# Options Spread Analysis

Run `uv run scripts/analyze.py` to analyze SPX conditions for put/call spread selling.

## Workflow

1. Run `uv run scripts/analyze.py`
2. Show the full script output to the user
3. Add a 2-3 sentence **Summary** covering:
   - The recommendation (which spread or wait)
   - Key risk factor (VIX level, upcoming events, or regime issue)
   - Next action (e.g., "re-check after CPI", "wait for dip", "clear to enter")

## Signals

| Signal | Meaning |
|--------|---------|
| `GREEN_LIGHT` | Good entry conditions |
| `GREEN_LIGHT (HIGH CONVICTION)` | Strong setup with multiple confirmations |
| `WAIT` | Regime OK but no entry signal yet |
| `RED_FLAG` | Don't trade — unfavorable regime or VIX > 18 |

## Options

```bash
uv run scripts/analyze.py --plot      # Show chart
uv run scripts/analyze.py --json      # JSON output
uv run scripts/analyze.py --synthetic # Test without API key
```

For detailed trading parameters and analysis logic, see [REFERENCE.md](REFERENCE.md).
