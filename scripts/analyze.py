#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "httpx",
#     "numpy",
#     "pandas",
#     "statsmodels",
#     "matplotlib",
# ]
# ///
"""
Options Spread Analysis - Put vs Call spread entry conditions.

Usage:
    uv run scripts/analyze.py              # Live data (requires FMP_API_KEY)
    uv run scripts/analyze.py --synthetic  # Test with synthetic data
    uv run scripts/analyze.py --plot       # Show plot
    uv run scripts/analyze.py --json       # Output as JSON

Requires: FMP_API_KEY environment variable
Get free key: https://financialmodelingprep.com/developer/docs/
"""

import json
import os
import sys
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta

import httpx
import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore", category=InterpolationWarning)

FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"


class FMPError(Exception):
    pass


def get_api_key() -> str:
    key = os.environ.get("FMP_API_KEY")
    if not key:
        raise FMPError("FMP_API_KEY not set. Get free key at https://financialmodelingprep.com")
    return key


def fetch_prices(symbol: str, days: int = 200, api_key: str | None = None) -> pd.DataFrame:
    """Fetch historical EOD prices from FMP."""
    if api_key is None:
        api_key = get_api_key()

    symbol_map = {"SPX": "^GSPC", "VIX": "^VIX"}
    fmp_symbol = symbol_map.get(symbol.upper(), symbol)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days + 30)

    url = f"{FMP_BASE_URL}/historical-price-full/{fmp_symbol}"
    params = {
        "apikey": api_key,
        "from": start_date.strftime("%Y-%m-%d"),
        "to": end_date.strftime("%Y-%m-%d"),
    }

    with httpx.Client(timeout=30.0) as client:
        response = client.get(url, params=params)
        response.raise_for_status()
        data = response.json()

    if "historical" not in data:
        raise FMPError(f"No data for {symbol}")

    df = pd.DataFrame(data["historical"])
    df = df.rename(columns={"date": "Date", "close": "Close", "open": "Open", "high": "High", "low": "Low"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").tail(days).reset_index(drop=True)
    return df


def fetch_quote(symbol: str, api_key: str | None = None) -> float:
    """Get current price."""
    if api_key is None:
        api_key = get_api_key()

    symbol_map = {"SPX": "^GSPC", "VIX": "^VIX"}
    fmp_symbol = symbol_map.get(symbol.upper(), symbol)

    url = f"{FMP_BASE_URL}/quote/{fmp_symbol}"
    with httpx.Client(timeout=30.0) as client:
        response = client.get(url, params={"apikey": api_key})
        response.raise_for_status()
        data = response.json()

    if not data:
        raise FMPError(f"No quote for {symbol}")
    return float(data[0]["price"])


def fetch_economic_calendar(days: int = 3, api_key: str | None = None) -> list[dict]:
    """Fetch upcoming high-impact economic events from FMP, grouped by day."""
    if api_key is None:
        api_key = get_api_key()

    today = datetime.now()
    end_date = today + timedelta(days=days)

    url = f"{FMP_BASE_URL}/economic_calendar"
    params = {
        "apikey": api_key,
        "from": today.strftime("%Y-%m-%d"),
        "to": end_date.strftime("%Y-%m-%d"),
    }

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

        # Group high-impact US events by date
        by_date: dict[str, list[str]] = {}
        for item in data:
            if item.get("country") != "US" or item.get("impact") != "High":
                continue
            date_part = item.get("date", "").split(" ")[0]
            event_name = item.get("event", "")
            if date_part not in by_date:
                by_date[date_part] = []
            by_date[date_part].append(event_name)

        # Format as one entry per day with comma-separated events
        result = []
        for date in sorted(by_date.keys()):
            events_str = ", ".join(by_date[date][:5])  # Limit to 5 per day
            if len(by_date[date]) > 5:
                events_str += f" (+{len(by_date[date]) - 5} more)"
            result.append({"date": date, "event": events_str, "impact": "high"})

        return result

    except Exception:
        return []  # Fail silently - events are supplementary


@dataclass
class MarketEvent:
    date: str
    event: str
    impact: str  # "high", "medium", "low"


@dataclass
class SpreadSignal:
    signal: str  # GREEN_LIGHT, WAIT, RED_FLAG
    zscore: float
    reasons: list[str]
    conviction_boosters: list[str]


@dataclass
class MarketAnalysis:
    timestamp: str
    spx_price: float
    vix_level: float
    vix_direction: str  # "rising", "falling", "flat"
    vix_change_5d: float  # 5-day VIX change %

    # Regime (applies to both)
    hurst: float
    hurst_regime: str
    variance_ratio: float
    regime_ok: bool

    # Half-life and DTE
    half_life_days: float
    recommended_dte_min: int
    recommended_dte_max: int

    # Put spread analysis
    put_signal: SpreadSignal

    # Call spread analysis
    call_signal: SpreadSignal

    # Recommendation
    recommendation: str

    # Upcoming events (next 3 days)
    events: list[MarketEvent]


def variance_ratio(prices: pd.Series, k: int) -> float:
    """Compute variance ratio for lag k."""
    log_prices = np.log(prices)
    returns_1 = log_prices.diff(1).dropna()
    returns_k = log_prices.diff(k).dropna()
    var_1 = returns_1.var(ddof=1)
    var_k = returns_k.var(ddof=1)
    return var_k / (k * var_1) if var_1 > 0 else 1.0


def calculate_hurst(prices: pd.Series, max_lag: int = 100) -> tuple[float, str]:
    """Calculate Hurst exponent using R/S analysis."""
    prices = prices.dropna().values
    n = len(prices)
    if n < 20:
        return 0.5, "RANDOM_WALK"

    returns = np.diff(np.log(prices))
    lags, rs_values = [], []

    for lag in range(10, min(max_lag, n // 2)):
        n_chunks = len(returns) // lag
        if n_chunks < 1:
            continue
        rs_chunk = []
        for i in range(n_chunks):
            chunk = returns[i * lag : (i + 1) * lag]
            mean_chunk = np.mean(chunk)
            cumdev = np.cumsum(chunk - mean_chunk)
            r = np.max(cumdev) - np.min(cumdev)
            s = np.std(chunk, ddof=1)
            if s > 0:
                rs_chunk.append(r / s)
        if rs_chunk:
            lags.append(lag)
            rs_values.append(np.mean(rs_chunk))

    if len(lags) < 3:
        return 0.5, "RANDOM_WALK"

    log_lags = np.log(lags)
    log_rs = np.log(rs_values)
    n_pts = len(log_lags)
    hurst = (n_pts * np.sum(log_lags * log_rs) - np.sum(log_lags) * np.sum(log_rs)) / (
        n_pts * np.sum(log_lags**2) - np.sum(log_lags) ** 2
    )
    hurst = max(0.0, min(1.0, hurst))

    if hurst < 0.45:
        regime = "MEAN_REVERTING"
    elif hurst < 0.55:
        regime = "RANDOM_WALK"
    else:
        regime = "TRENDING"

    return hurst, regime


def calculate_half_life(stretch: pd.Series) -> float:
    """Calculate half-life from Ornstein-Uhlenbeck model."""
    s = stretch.dropna().values
    if len(s) < 10:
        return float("inf")

    y = np.diff(s)
    x = s[:-1]
    var_x = np.var(x)
    if var_x == 0:
        return float("inf")

    lambda_param = np.cov(x, y)[0, 1] / var_x
    if lambda_param >= 0:
        return float("inf")

    ar_coef = 1 + lambda_param
    if ar_coef <= 0:
        return float("inf")

    return -np.log(2) / np.log(ar_coef)


def get_vix_direction(vix_series: pd.Series, lookback: int = 5) -> tuple[str, float]:
    """Determine VIX direction based on recent trend."""
    if len(vix_series) < lookback + 1:
        return "flat", 0.0

    recent = vix_series.tail(lookback + 1)
    change_pct = (recent.iloc[-1] - recent.iloc[0]) / recent.iloc[0] * 100

    if change_pct > 5:
        return "rising", change_pct
    elif change_pct < -5:
        return "falling", change_pct
    else:
        return "flat", change_pct


def analyze_put_spread(
    zscore: float,
    vix_direction: str,
    vix_zscore: float,
    regime_ok: bool,
    stationarity_ok: bool,
    half_life: float,
) -> SpreadSignal:
    """Analyze conditions for bull put spread (sell put, buy lower put)."""
    reasons = []
    boosters = []

    if not regime_ok:
        return SpreadSignal(
            signal="RED_FLAG",
            zscore=zscore,
            reasons=["Unfavorable regime - trending or momentum market"],
            conviction_boosters=[],
        )

    # Check for dip
    dip_ok = zscore < -1.0
    vix_ok = vix_direction in ("rising", "flat") or vix_zscore > 0

    if dip_ok:
        reasons.append(f"Dip detected (Z={zscore:.2f})")
    else:
        reasons.append(f"No significant dip (Z={zscore:.2f})")

    if vix_ok:
        reasons.append(f"VIX {vix_direction} - put premiums favorable")
    else:
        reasons.append(f"VIX declining - put premiums may be thin")

    # Conviction boosters
    if zscore < -2.0:
        boosters.append("deep dip")
    if vix_zscore > 2.0:
        boosters.append("VIX spike")
    if stationarity_ok:
        boosters.append("stationarity confirmed")
    if half_life < 10:
        boosters.append("fast reversion")

    # Determine signal
    if dip_ok and vix_ok:
        if len(boosters) >= 2:
            signal = "GREEN_LIGHT (HIGH CONVICTION)"
        else:
            signal = "GREEN_LIGHT"
    elif dip_ok or vix_ok:
        signal = "WAIT"
    else:
        signal = "WAIT"

    return SpreadSignal(
        signal=signal,
        zscore=zscore,
        reasons=reasons,
        conviction_boosters=boosters,
    )


def analyze_call_spread(
    zscore: float,
    vix_direction: str,
    vix_zscore: float,
    regime_ok: bool,
    stationarity_ok: bool,
    half_life: float,
) -> SpreadSignal:
    """Analyze conditions for bear call spread (sell call, buy higher call)."""
    reasons = []
    boosters = []

    if not regime_ok:
        return SpreadSignal(
            signal="RED_FLAG",
            zscore=zscore,
            reasons=["Unfavorable regime - trending or momentum market"],
            conviction_boosters=[],
        )

    # Check for rally (mirror of dip logic)
    rally_ok = zscore > 1.0
    vix_ok = vix_direction == "falling"

    if rally_ok:
        reasons.append(f"Rally detected (Z={zscore:.2f})")
    else:
        reasons.append(f"No significant rally (Z={zscore:.2f})")

    if vix_ok:
        reasons.append(f"VIX declining - confirms rally")
    else:
        reasons.append(f"VIX {vix_direction} - rally may not sustain")

    # Conviction boosters
    if zscore > 2.0:
        boosters.append("strong rally")
    if vix_zscore < -1.0:
        boosters.append("VIX crushed")
    if stationarity_ok:
        boosters.append("stationarity confirmed")
    if half_life < 10:
        boosters.append("fast reversion")

    # Determine signal
    if rally_ok and vix_ok:
        if len(boosters) >= 2:
            signal = "GREEN_LIGHT (HIGH CONVICTION)"
        else:
            signal = "GREEN_LIGHT"
    elif rally_ok or vix_ok:
        signal = "WAIT"
    else:
        signal = "WAIT"

    return SpreadSignal(
        signal=signal,
        zscore=zscore,
        reasons=reasons,
        conviction_boosters=boosters,
    )


def analyze_market(
    spx_df: pd.DataFrame,
    current_spx: float,
    vix_series: pd.Series,
    current_vix: float,
    events: list[dict] | None = None,
    ma_period: int = 20,
) -> MarketAnalysis:
    """Run complete market analysis for both put and call spreads."""
    closes = spx_df["Close"]

    # Calculate stretch
    ma = closes.rolling(ma_period).mean()
    stretch = closes - ma
    stretch_clean = stretch.dropna()

    current_ma = closes.tail(ma_period).mean()
    current_stretch = current_spx - current_ma

    # Z-Score
    rolling_std = stretch_clean.tail(ma_period).std()
    zscore = current_stretch / rolling_std if rolling_std > 0 else 0

    # Hurst exponent
    hurst, hurst_regime = calculate_hurst(closes.tail(100))

    # Variance ratio (recent regime)
    vr = variance_ratio(closes.tail(35), 5)

    # Half-life
    hl = calculate_half_life(stretch_clean)
    if 0 < hl < 100:
        dte_min = max(14, int(np.ceil(hl)))
        dte_max = min(35, max(28, int(np.ceil(2 * hl))))
    else:
        dte_min, dte_max = 21, 28

    # Stationarity tests
    adf_p = adfuller(stretch_clean, autolag="AIC")[1]
    kpss_p = kpss(stretch_clean, regression="c", nlags="auto")[1]
    stationarity_ok = adf_p < 0.05 and kpss_p > 0.05

    # VIX analysis
    vix_direction, vix_change = get_vix_direction(vix_series)
    vix_ma = vix_series.rolling(ma_period).mean()
    vix_stretch = vix_series - vix_ma
    vix_std = vix_stretch.tail(ma_period).std()
    vix_zscore = (current_vix - vix_series.tail(ma_period).mean()) / vix_std if vix_std > 0 else 0

    # Regime check (applies to both directions)
    hurst_ok = hurst < 0.5
    vr_ok = vr <= 1.0
    regime_ok = hurst_ok and vr_ok

    # Analyze both directions
    put_signal = analyze_put_spread(zscore, vix_direction, vix_zscore, regime_ok, stationarity_ok, hl)
    call_signal = analyze_call_spread(zscore, vix_direction, vix_zscore, regime_ok, stationarity_ok, hl)

    # Generate recommendation
    if not regime_ok:
        recommendation = "NO TRADE - unfavorable regime"
    elif "GREEN_LIGHT" in put_signal.signal and "GREEN_LIGHT" in call_signal.signal:
        # Both look good - pick stronger
        if abs(zscore) > 2:
            if zscore < 0:
                recommendation = "PUT SPREAD preferred (deeper dip)"
            else:
                recommendation = "CALL SPREAD preferred (stronger rally)"
        else:
            recommendation = "Either side acceptable - pick based on bias"
    elif "GREEN_LIGHT" in put_signal.signal:
        recommendation = "Consider PUT SPREAD"
    elif "GREEN_LIGHT" in call_signal.signal:
        recommendation = "Consider CALL SPREAD"
    elif put_signal.signal == "WAIT" or call_signal.signal == "WAIT":
        recommendation = "WAIT for better entry"
    else:
        recommendation = "NO TRADE today"

    # Convert events to MarketEvent objects
    event_list = []
    if events:
        for e in events:
            event_list.append(MarketEvent(
                date=e["date"],
                event=e["event"],
                impact=e["impact"],
            ))

    return MarketAnalysis(
        timestamp=datetime.now().isoformat(),
        spx_price=current_spx,
        vix_level=current_vix,
        vix_direction=vix_direction,
        vix_change_5d=round(vix_change, 2),
        hurst=round(hurst, 4),
        hurst_regime=hurst_regime,
        variance_ratio=round(vr, 4),
        regime_ok=regime_ok,
        half_life_days=round(hl, 1) if hl < 100 else -1,
        recommended_dte_min=dte_min,
        recommended_dte_max=dte_max,
        put_signal=put_signal,
        call_signal=call_signal,
        recommendation=recommendation,
        events=event_list,
    )


def generate_synthetic_data(n: int = 150, seed: int = 42, scenario: str = "dip"):
    """Generate synthetic SPX and VIX data for testing."""
    np.random.seed(seed)

    # Mean-reverting SPX
    base = 5800
    trend = np.linspace(0, 100, n)
    mr = np.zeros(n)
    for i in range(1, n):
        mr[i] = 0.75 * mr[i - 1] + np.random.normal(0, 20)
    prices = base + trend + mr

    dates = pd.date_range(end=datetime.now(), periods=n, freq="D")
    spx_df = pd.DataFrame({
        "Date": dates,
        "Open": prices - np.random.uniform(0, 5, n),
        "High": prices + np.random.uniform(5, 15, n),
        "Low": prices - np.random.uniform(5, 15, n),
        "Close": prices,
    })

    # Mean-reverting VIX
    vix = np.zeros(n)
    vix[0] = 18
    for i in range(1, n):
        shock = np.random.normal(0, 2) + (10 if np.random.random() > 0.95 else 0)
        vix[i] = max(10, vix[i - 1] + 0.1 * (18 - vix[i - 1]) + shock)

    # Simulate scenario
    stretch_std = (pd.Series(prices) - pd.Series(prices).rolling(20).mean()).std()

    if scenario == "dip":
        current_spx = prices[-1] - 2.5 * stretch_std
        current_vix = vix[-1] + 5
        # Make VIX rising
        vix[-5:] = vix[-5:] + np.linspace(0, 3, 5)
    elif scenario == "rally":
        current_spx = prices[-1] + 2.5 * stretch_std
        current_vix = vix[-1] - 3
        # Make VIX falling
        vix[-5:] = vix[-5:] - np.linspace(0, 3, 5)
    else:  # neutral
        current_spx = prices[-1]
        current_vix = vix[-1]

    return spx_df, pd.Series(vix), current_spx, current_vix


def print_analysis(analysis: MarketAnalysis):
    """Print formatted analysis to stdout."""
    # Colors
    green = "\033[92m"
    yellow = "\033[93m"
    red = "\033[91m"
    cyan = "\033[96m"
    reset = "\033[0m"
    bold = "\033[1m"

    def signal_color(sig):
        if "GREEN_LIGHT" in sig:
            return green
        elif sig == "WAIT":
            return yellow
        return red

    print(f"\n{bold}{'═' * 65}")
    print(f" OPTIONS SPREAD ANALYSIS")
    print(f"{'═' * 65}{reset}")

    # Market data
    vix_arrow = "↑" if analysis.vix_direction == "rising" else "↓" if analysis.vix_direction == "falling" else "→"
    print(f"\n {cyan}SPX:{reset} {analysis.spx_price:,.2f}  |  {cyan}VIX:{reset} {analysis.vix_level:.2f} {vix_arrow} ({analysis.vix_change_5d:+.1f}% 5d)")

    # Regime
    regime_status = f"{green}OK{reset}" if analysis.regime_ok else f"{red}UNFAVORABLE{reset}"
    print(f" {cyan}Regime:{reset} {regime_status} (Hurst={analysis.hurst:.3f} {analysis.hurst_regime}, VR={analysis.variance_ratio:.3f})")
    print(f" {cyan}DTE:{reset} {analysis.recommended_dte_min}-{analysis.recommended_dte_max} days (half-life={analysis.half_life_days:.1f}d)")

    # Side by side comparison
    print(f"\n {bold}PUT SPREAD                    CALL SPREAD{reset}")
    print(f" {'─' * 28}  {'─' * 28}")

    put_c = signal_color(analysis.put_signal.signal)
    call_c = signal_color(analysis.call_signal.signal)

    print(f" Signal: {put_c}{analysis.put_signal.signal:<20}{reset}  Signal: {call_c}{analysis.call_signal.signal}{reset}")
    print(f" Z-Score: {analysis.put_signal.zscore:<19.2f}  Z-Score: {analysis.call_signal.zscore:+.2f}")

    # Reasons
    print(f"\n {cyan}Put Reasons:{reset}")
    for r in analysis.put_signal.reasons:
        print(f"   • {r}")
    if analysis.put_signal.conviction_boosters:
        print(f"   {green}Boosters: {', '.join(analysis.put_signal.conviction_boosters)}{reset}")

    print(f"\n {cyan}Call Reasons:{reset}")
    for r in analysis.call_signal.reasons:
        print(f"   • {r}")
    if analysis.call_signal.conviction_boosters:
        print(f"   {green}Boosters: {', '.join(analysis.call_signal.conviction_boosters)}{reset}")

    # Upcoming events
    if analysis.events:
        print(f"\n {cyan}Upcoming Events (next 3 days):{reset}")
        for evt in analysis.events:
            evt_color = red if evt.impact == "high" else yellow if evt.impact == "medium" else reset
            print(f"   {evt_color}• {evt.date}: {evt.event} [{evt.impact.upper()}]{reset}")
    else:
        print(f"\n {cyan}Upcoming Events:{reset} None significant in next 3 days")

    # Recommendation
    rec_color = green if "Consider" in analysis.recommendation else yellow if "WAIT" in analysis.recommendation else red
    print(f"\n {bold}► RECOMMENDATION:{reset} {rec_color}{analysis.recommendation}{reset}")

    print(f"\n{'═' * 65}\n")


def plot_analysis(
    spx_df: pd.DataFrame,
    vix_series: pd.Series,
    analysis: MarketAnalysis,
    save_path: str | None = None,
    show: bool = True,
):
    """Plot analysis with put/call zones."""
    import matplotlib.pyplot as plt

    closes = spx_df["Close"]
    dates = spx_df["Date"]
    ma = closes.rolling(20).mean()
    stretch = closes - ma
    stretch_std = stretch.rolling(20).std()
    zscore_series = stretch / stretch_std

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    green = "#22c55e"
    red = "#ef4444"
    blue = "#3b82f6"
    gray = "#6b7280"

    # Plot 1: Price
    ax1 = axes[0]
    ax1.plot(dates, closes, color=blue, linewidth=1.5, label="SPX")
    ax1.plot(dates, ma, color=gray, linestyle="--", label="MA(20)")
    ax1.scatter([dates.iloc[-1]], [analysis.spx_price], color=blue, s=100, zorder=5)
    ax1.set_ylabel("SPX Price")
    ax1.set_title(f"Options Spread Analysis | Recommendation: {analysis.recommendation}")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Z-Score with zones
    ax2 = axes[1]
    ax2.fill_between(dates, 1, 3, color=red, alpha=0.15, label="Call spread zone (Z > 1)")
    ax2.fill_between(dates, -3, -1, color=green, alpha=0.15, label="Put spread zone (Z < -1)")
    ax2.plot(dates, zscore_series, color=blue, linewidth=1.5)
    ax2.axhline(0, color=gray, linewidth=0.5)
    ax2.axhline(1, color=red, linewidth=1, linestyle="--", alpha=0.5)
    ax2.axhline(-1, color=green, linewidth=1, linestyle="--", alpha=0.5)

    current_z = analysis.put_signal.zscore
    z_color = green if current_z < -1 else red if current_z > 1 else gray
    ax2.scatter([dates.iloc[-1]], [current_z], color=z_color, s=100, zorder=5)
    ax2.annotate(f"Z={current_z:.2f}", xy=(dates.iloc[-1], current_z), xytext=(10, 0),
                 textcoords="offset points", fontsize=9, color=z_color, fontweight="bold")

    ax2.set_ylabel("Z-Score")
    ax2.set_ylim(-4, 4)
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    # Plot 3: VIX
    ax3 = axes[2]
    vix_dates = dates.iloc[-len(vix_series):]
    ax3.plot(vix_dates, vix_series.values, color="#f59e0b", linewidth=1.5, label="VIX")
    ax3.axhline(20, color=gray, linestyle=":", linewidth=0.5)

    vix_color = red if analysis.vix_direction == "rising" else green if analysis.vix_direction == "falling" else gray
    ax3.scatter([vix_dates.iloc[-1]], [analysis.vix_level], color=vix_color, s=100, zorder=5)

    arrow = "↑" if analysis.vix_direction == "rising" else "↓" if analysis.vix_direction == "falling" else "→"
    ax3.annotate(f"VIX={analysis.vix_level:.1f} {arrow}", xy=(vix_dates.iloc[-1], analysis.vix_level),
                 xytext=(10, 0), textcoords="offset points", fontsize=9, color=vix_color, fontweight="bold")

    ax3.set_ylabel("VIX")
    ax3.set_xlabel("Date")
    ax3.legend(loc="upper left")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved: {save_path}")

    if show:
        plt.show()

    return fig


def main():
    use_synthetic = "--synthetic" in sys.argv or "-s" in sys.argv
    output_json = "--json" in sys.argv or "-j" in sys.argv
    show_plot = "--plot" in sys.argv or "-p" in sys.argv
    save_plot = None
    for arg in sys.argv:
        if arg.startswith("--save="):
            save_plot = arg.split("=", 1)[1]

    # Scenario for synthetic data
    scenario = "dip"  # default
    for arg in sys.argv:
        if arg.startswith("--scenario="):
            scenario = arg.split("=", 1)[1]

    try:
        events = []
        if use_synthetic:
            spx_df, vix_series, current_spx, current_vix = generate_synthetic_data(scenario=scenario)
            if not output_json:
                print(f"Using synthetic data (scenario: {scenario})...")
        else:
            api_key = get_api_key()
            if not output_json:
                print("Fetching live data...", end=" ", flush=True)

            spx_df = fetch_prices("^GSPC", days=150, api_key=api_key)
            vix_df = fetch_prices("^VIX", days=150, api_key=api_key)

            # Align VIX to SPX dates
            vix_df = vix_df.set_index("Date")
            spx_df_indexed = spx_df.set_index("Date")
            vix_series = vix_df["Close"].reindex(spx_df_indexed.index).ffill().reset_index(drop=True)

            current_spx = fetch_quote("^GSPC", api_key)
            current_vix = fetch_quote("^VIX", api_key)

            # Fetch economic calendar
            events = fetch_economic_calendar(days=3, api_key=api_key)

            if not output_json:
                print("done.")

        analysis = analyze_market(spx_df, current_spx, vix_series, current_vix, events=events)

        if output_json:
            # Convert dataclass to dict for JSON
            data = asdict(analysis)
            print(json.dumps(data, indent=2))
        else:
            print_analysis(analysis)

            if show_plot or save_plot:
                plot_analysis(spx_df, vix_series, analysis, save_path=save_plot, show=show_plot)

    except FMPError as e:
        if output_json:
            print(json.dumps({"error": str(e)}))
        else:
            print(f"\nError: {e}")
            print("\nRun with --synthetic for testing without API key.")
        sys.exit(1)
    except Exception as e:
        if output_json:
            print(json.dumps({"error": str(e)}))
        else:
            print(f"\nError: {e}")
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()
