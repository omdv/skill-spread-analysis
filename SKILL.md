---
name: options-spread-analysis
description: Analyzes market conditions for selling option spreads (put or call). Use when the user asks whether to open a bull put spread or bear call spread, wants a market check before trading, or asks which direction looks better for premium selling.
---

# Options Spread Analysis

Run `uv run scripts/analyze.py` to analyze SPX conditions for put/call spread selling.

## Workflow

1. Run `uv run scripts/analyze.py`
2. Format the results as a clean text report (see format below)
3. Add a 2-3 sentence **Summary** at the end

## Output Format

```
Options Spread Analysis Report (Live Data - [DATE TIME] UTC)

 SPX: [price]  |  VIX: [level] [arrow] ([change]% 5d)
 Regime: [OK/UNFAVORABLE] (Hurst=[H] [regime], VR=[vr])
 DTE: [min]-[max] days (half-life=[hl]d)

PUT SPREAD                    CALL SPREAD
───────────────────────────   ───────────────────────────
 Signal: [signal]             Signal: [signal]
 Z-Score: [z]                 Z-Score: [z]

Put Reasons:
  • [reason 1]
  • [reason 2]
  Boosters: [if any]

Call Reasons:
  • [reason 1]
  • [reason 2]
  Boosters: [if any]

Upcoming Events (next 3 days):
  • [date]: [event] [HIGH]
  [or "None" if clear]

► RECOMMENDATION: [recommendation]

Summary:
[2-3 sentences: recommendation, key risk, next action]
```

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
