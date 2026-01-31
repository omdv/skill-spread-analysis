import os
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta

import httpx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss

# Suppress KPSS interpolation warnings (expected when statistic is outside lookup table range)
warnings.filterwarnings("ignore", category=InterpolationWarning)

# Financial Modeling Prep API base URL
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"


class FMPError(Exception):
    """Exception raised for Financial Modeling Prep API errors."""

    pass


def get_fmp_api_key() -> str:
    """Get FMP API key from environment variable."""
    api_key = os.environ.get("FMP_API_KEY")
    if not api_key:
        raise FMPError(
            "FMP_API_KEY environment variable not set. "
            "Get a free API key at https://financialmodelingprep.com/developer/docs/"
        )
    return api_key


def fetch_historical_prices(
    symbol: str,
    days: int = 200,
    api_key: str | None = None,
) -> pd.DataFrame:
    """
    Fetch historical EOD prices from Financial Modeling Prep.

    Args:
        symbol: Stock/index symbol (e.g., "SPY", "^GSPC", "^VIX")
        days: Number of historical days to fetch
        api_key: FMP API key (uses FMP_API_KEY env var if not provided)

    Returns:
        DataFrame with columns: Date, Open, High, Low, Close, Volume
        Sorted by date ascending (oldest first)
    """
    if api_key is None:
        api_key = get_fmp_api_key()

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days + 30)  # Extra buffer for weekends/holidays

    url = f"{FMP_BASE_URL}/historical-price-full/{symbol}"
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
        raise FMPError(f"No historical data found for symbol: {symbol}. Response: {data}")

    # Convert to DataFrame
    df = pd.DataFrame(data["historical"])

    # Rename columns to standard OHLC format
    df = df.rename(
        columns={
            "date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )

    # Convert date and sort ascending
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date", ascending=True).reset_index(drop=True)

    # Keep only the requested number of days
    df = df.tail(days).reset_index(drop=True)

    return df[["Date", "Open", "High", "Low", "Close", "Volume"]]


def fetch_index_prices(
    symbol: str = "^GSPC",
    days: int = 200,
    api_key: str | None = None,
) -> pd.DataFrame:
    """
    Fetch historical prices for an index (SPX, VIX, etc.).

    Common symbols:
        - "^GSPC" or "SPX": S&P 500 Index
        - "^VIX" or "VIX": CBOE Volatility Index
        - "SPY": S&P 500 ETF
        - "QQQ": Nasdaq 100 ETF

    Args:
        symbol: Index symbol
        days: Number of historical days
        api_key: FMP API key

    Returns:
        DataFrame with OHLC data
    """
    # FMP uses different symbols for some indices
    symbol_map = {
        "SPX": "^GSPC",
        "VIX": "^VIX",
    }
    fmp_symbol = symbol_map.get(symbol.upper(), symbol)

    return fetch_historical_prices(fmp_symbol, days, api_key)


def fetch_spx_and_vix(
    days: int = 200,
    api_key: str | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Fetch both SPX and VIX data for mean reversion analysis.

    Args:
        days: Number of historical days
        api_key: FMP API key

    Returns:
        Tuple of (spx_df, vix_series) where:
            - spx_df: DataFrame with SPX OHLC data
            - vix_series: Series with VIX close prices (aligned to SPX dates)
    """
    if api_key is None:
        api_key = get_fmp_api_key()

    # Fetch both datasets
    spx_df = fetch_index_prices("^GSPC", days, api_key)
    vix_df = fetch_index_prices("^VIX", days, api_key)

    # Align VIX to SPX dates
    vix_df = vix_df.set_index("Date")
    spx_df = spx_df.set_index("Date")

    # Get VIX closes aligned to SPX dates
    vix_series = vix_df["Close"].reindex(spx_df.index).ffill()

    # Reset index for SPX
    spx_df = spx_df.reset_index()

    return spx_df, vix_series.reset_index(drop=True)


def get_current_price(
    symbol: str,
    api_key: str | None = None,
) -> float:
    """
    Fetch current/latest price for a symbol.

    Args:
        symbol: Stock/index symbol
        api_key: FMP API key

    Returns:
        Current price as float
    """
    if api_key is None:
        api_key = get_fmp_api_key()

    # FMP symbol mapping
    symbol_map = {
        "SPX": "^GSPC",
        "VIX": "^VIX",
    }
    fmp_symbol = symbol_map.get(symbol.upper(), symbol)

    url = f"{FMP_BASE_URL}/quote/{fmp_symbol}"
    params = {"apikey": api_key}

    with httpx.Client(timeout=30.0) as client:
        response = client.get(url, params=params)
        response.raise_for_status()
        data = response.json()

    if not data:
        raise FMPError(f"No quote data found for symbol: {symbol}")

    return float(data[0]["price"])


@dataclass
class StretchTestResult:
    adf_statistic: float
    adf_pvalue: float
    adf_is_stationary: bool  # True if p < 0.05
    kpss_statistic: float
    kpss_pvalue: float
    kpss_is_stationary: bool  # True if p > 0.05
    confirmed_mean_reverting: bool  # Both tests agree


@dataclass
class VarianceRatioResult:
    lag: int
    ratio: float
    interpretation: str  # "mean_reverting" | "random_walk" | "momentum"


@dataclass
class ZScoreResult:
    """Z-Score normalized stretch analysis."""

    current_zscore: float  # Current stretch in standard deviations
    zscore_threshold: float  # Threshold for significance (default -2.0)
    is_extreme_dip: bool  # True if zscore < threshold
    rolling_std: float  # Current rolling volatility used for normalization


@dataclass
class HalfLifeResult:
    """Ornstein-Uhlenbeck half-life calculation."""

    lambda_param: float  # Mean reversion speed parameter
    half_life_days: float  # Days to recover halfway to mean
    recommended_min_dte: int  # Minimum DTE (1x half-life)
    recommended_max_dte: int  # Maximum DTE (2x half-life)
    is_valid: bool  # True if lambda > 0 (valid mean reversion)


@dataclass
class HurstResult:
    """Hurst exponent regime classification."""

    hurst_exponent: float  # H value between 0 and 1
    regime: str  # "MEAN_REVERTING" | "RANDOM_WALK" | "TRENDING"
    is_safe_for_puts: bool  # True if H < 0.4


@dataclass
class VIXAnalysis:
    """VIX mean reversion analysis for volatility trading."""

    current_vix: float
    vix_zscore: float
    vix_ma: float
    vix_is_elevated: bool  # True if zscore > threshold (premiums rich)
    vix_stretch_test: StretchTestResult
    vix_confirmed_mean_reverting: bool


@dataclass
class MeanReversionAnalysis:
    # Stretch test results (validates strategy thesis)
    stretch_test: StretchTestResult

    # VR at multiple lags (for DTE selection)
    variance_ratios: list[VarianceRatioResult]
    optimal_dte_lag: int | None  # Lag with lowest VR < 1 (None if no mean reversion found)

    # Rolling VR (regime detection)
    current_regime_vr: float
    regime: str  # "MEAN_REVERTING" | "MOMENTUM"

    # Current stretch info (for live trading)
    current_stretch: float  # Current price - MA
    current_stretch_pct: float  # As percentage of MA

    # === NEW: Enhanced Analysis ===

    # Z-Score normalized analysis
    zscore: ZScoreResult

    # Half-life from Ornstein-Uhlenbeck
    half_life: HalfLifeResult

    # Hurst exponent regime filter
    hurst: HurstResult

    # VIX analysis (optional, None if no VIX data provided)
    vix_analysis: VIXAnalysis | None

    # Final signal
    signal: str  # "GREEN_LIGHT" | "RED_FLAG"
    signal_reasons: list[str]  # Explanation of signal


def variance_ratio(prices: pd.Series, k: int) -> float:
    """
    Compute variance ratio for lag k using log returns.

    Using log returns normalizes for price level - a 50-point move at SPX 2000
    is different from a 50-point move at SPX 5800.
    """
    log_prices = np.log(prices)
    returns_1 = log_prices.diff(1).dropna()
    returns_k = log_prices.diff(k).dropna()

    var_1 = returns_1.var(ddof=1)
    var_k = returns_k.var(ddof=1)

    if var_1 == 0:
        return 1.0

    return var_k / (k * var_1)


def calculate_zscore(
    stretch_series: pd.Series,
    current_stretch: float,
    lookback: int = 20,
) -> ZScoreResult:
    """Calculate Z-Score of current stretch relative to recent volatility."""
    rolling_std = stretch_series.tail(lookback).std()
    if rolling_std == 0:
        rolling_std = 1e-10  # Avoid division by zero

    current_zscore = current_stretch / rolling_std
    zscore_threshold = -2.0
    is_extreme_dip = current_zscore < zscore_threshold

    return ZScoreResult(
        current_zscore=current_zscore,
        zscore_threshold=zscore_threshold,
        is_extreme_dip=is_extreme_dip,
        rolling_std=rolling_std,
    )


def calculate_half_life(stretch_series: pd.Series) -> HalfLifeResult:
    """
    Calculate half-life using Ornstein-Uhlenbeck model.

    Uses OLS regression: delta_stretch = lambda * stretch + epsilon
    Half-life = -ln(2) / ln(1 + lambda)
    """
    stretch = stretch_series.dropna().values

    # Lagged regression: stretch[t] - stretch[t-1] = lambda * stretch[t-1] + error
    y = np.diff(stretch)  # delta_stretch
    x = stretch[:-1]  # lagged stretch

    if len(x) < 10:
        return HalfLifeResult(
            lambda_param=0.0,
            half_life_days=float("inf"),
            recommended_min_dte=21,
            recommended_max_dte=45,
            is_valid=False,
        )

    # OLS: y = lambda * x + error
    # lambda = cov(x, y) / var(x)
    cov_xy = np.cov(x, y)[0, 1]
    var_x = np.var(x)

    if var_x == 0:
        lambda_param = 0.0
    else:
        lambda_param = cov_xy / var_x

    # Half-life calculation
    # The discrete AR(1) coefficient is (1 + lambda)
    # Half-life = -ln(2) / ln(1 + lambda)
    if lambda_param >= 0:
        # No mean reversion (lambda must be negative for mean reversion)
        half_life_days = float("inf")
        is_valid = False
    else:
        ar_coef = 1 + lambda_param
        if ar_coef <= 0:
            half_life_days = float("inf")
            is_valid = False
        else:
            half_life_days = -np.log(2) / np.log(ar_coef)
            is_valid = half_life_days > 0 and half_life_days < 100

    # Recommended DTE range
    if is_valid and half_life_days < 100:
        recommended_min_dte = max(1, int(np.ceil(half_life_days)))
        recommended_max_dte = max(2, int(np.ceil(2 * half_life_days)))
    else:
        recommended_min_dte = 21
        recommended_max_dte = 45

    return HalfLifeResult(
        lambda_param=lambda_param,
        half_life_days=half_life_days,
        recommended_min_dte=recommended_min_dte,
        recommended_max_dte=recommended_max_dte,
        is_valid=is_valid,
    )


def calculate_hurst_exponent(prices: pd.Series, max_lag: int = 100) -> HurstResult:
    """
    Calculate Hurst exponent using R/S (Rescaled Range) analysis.

    H < 0.5: Mean reverting
    H = 0.5: Random walk
    H > 0.5: Trending/momentum
    """
    prices = prices.dropna().values
    n = len(prices)

    if n < 20:
        return HurstResult(
            hurst_exponent=0.5,
            regime="RANDOM_WALK",
            is_safe_for_puts=False,
        )

    # Use log returns for Hurst calculation
    returns = np.diff(np.log(prices))

    # Calculate R/S for different lag sizes
    lags = []
    rs_values = []

    for lag in range(10, min(max_lag, n // 2)):
        # Split returns into non-overlapping chunks of size 'lag'
        n_chunks = len(returns) // lag
        if n_chunks < 1:
            continue

        rs_chunk = []
        for i in range(n_chunks):
            chunk = returns[i * lag : (i + 1) * lag]
            mean_chunk = np.mean(chunk)
            cumdev = np.cumsum(chunk - mean_chunk)
            r = np.max(cumdev) - np.min(cumdev)  # Range
            s = np.std(chunk, ddof=1)  # Standard deviation
            if s > 0:
                rs_chunk.append(r / s)

        if rs_chunk:
            lags.append(lag)
            rs_values.append(np.mean(rs_chunk))

    if len(lags) < 3:
        return HurstResult(
            hurst_exponent=0.5,
            regime="RANDOM_WALK",
            is_safe_for_puts=False,
        )

    # Linear regression: log(R/S) = H * log(lag) + c
    log_lags = np.log(lags)
    log_rs = np.log(rs_values)

    # OLS for Hurst exponent
    n_pts = len(log_lags)
    sum_x = np.sum(log_lags)
    sum_y = np.sum(log_rs)
    sum_xy = np.sum(log_lags * log_rs)
    sum_xx = np.sum(log_lags * log_lags)

    hurst = (n_pts * sum_xy - sum_x * sum_y) / (n_pts * sum_xx - sum_x * sum_x)

    # Clamp to valid range
    hurst = max(0.0, min(1.0, hurst))

    # Classify regime
    if hurst < 0.45:
        regime = "MEAN_REVERTING"
        is_safe = True
    elif hurst < 0.55:
        regime = "RANDOM_WALK"
        is_safe = True  # Still tradeable, just less conviction
    else:
        regime = "TRENDING"
        is_safe = False

    return HurstResult(
        hurst_exponent=hurst,
        regime=regime,
        is_safe_for_puts=is_safe,
    )


def analyze_vix(
    vix_prices: pd.Series,
    current_vix: float,
    ma_period: int = 20,
    zscore_threshold: float = 2.0,
    adf_pvalue_threshold: float = 0.05,
    kpss_pvalue_threshold: float = 0.05,
) -> VIXAnalysis:
    """Analyze VIX mean reversion for volatility trading edge."""
    # Calculate VIX stretch
    vix_ma = vix_prices.rolling(ma_period).mean()
    vix_stretch = vix_prices - vix_ma
    vix_stretch_clean = vix_stretch.dropna()

    # Current VIX stats
    current_vix_ma = vix_prices.tail(ma_period).mean()
    current_vix_stretch = current_vix - current_vix_ma

    # Z-Score for VIX
    vix_std = vix_stretch_clean.tail(ma_period).std()
    if vix_std == 0:
        vix_std = 1e-10
    vix_zscore = current_vix_stretch / vix_std

    # VIX is elevated if zscore > threshold (premiums are rich)
    vix_is_elevated = vix_zscore > zscore_threshold

    # Run stationarity tests on VIX stretch
    adf_result = adfuller(vix_stretch_clean, autolag="AIC")
    adf_statistic = adf_result[0]
    adf_pvalue = adf_result[1]
    adf_is_stationary = adf_pvalue < adf_pvalue_threshold

    kpss_result = kpss(vix_stretch_clean, regression="c", nlags="auto")
    kpss_statistic = kpss_result[0]
    kpss_pvalue = kpss_result[1]
    kpss_is_stationary = kpss_pvalue > kpss_pvalue_threshold

    confirmed_mean_reverting = adf_is_stationary and kpss_is_stationary

    vix_stretch_test = StretchTestResult(
        adf_statistic=adf_statistic,
        adf_pvalue=adf_pvalue,
        adf_is_stationary=adf_is_stationary,
        kpss_statistic=kpss_statistic,
        kpss_pvalue=kpss_pvalue,
        kpss_is_stationary=kpss_is_stationary,
        confirmed_mean_reverting=confirmed_mean_reverting,
    )

    return VIXAnalysis(
        current_vix=current_vix,
        vix_zscore=vix_zscore,
        vix_ma=current_vix_ma,
        vix_is_elevated=vix_is_elevated,
        vix_stretch_test=vix_stretch_test,
        vix_confirmed_mean_reverting=confirmed_mean_reverting,
    )


def analyze_mean_reversion(
    historical_prices: pd.DataFrame,
    current_price: float,
    price_column: str = "Close",
    ma_period: int = 20,
    vr_lags: list[int] | None = None,
    rolling_window: int = 30,
    adf_pvalue_threshold: float = 0.05,
    kpss_pvalue_threshold: float = 0.05,
    # New parameters for enhanced analysis
    zscore_threshold: float = -1.0,  # Relaxed from -2.0 for more frequent signals
    hurst_lookback: int = 100,
    hurst_threshold: float = 0.5,  # 0.5 = random walk; lower = more mean-reverting
    # Optional VIX data for volatility analysis
    vix_prices: pd.Series | None = None,
    current_vix: float | None = None,
    vix_zscore_threshold: float = 2.0,
) -> MeanReversionAnalysis:
    """
    Analyze mean reversion characteristics of price data for put-selling decisions.

    Args:
        historical_prices: DataFrame with OHLC data
        current_price: Today's current price or open price
        price_column: Column name for prices (default "Close")
        ma_period: Moving average period for stretch calculation
        vr_lags: List of lags for variance ratio tests (default [2, 3, 5, 10, 21])
        rolling_window: Window size for rolling variance ratio
        adf_pvalue_threshold: P-value threshold for ADF test
        kpss_pvalue_threshold: P-value threshold for KPSS test
        zscore_threshold: Z-score threshold for extreme dip detection (default -2.0)
        hurst_lookback: Lookback period for Hurst exponent calculation
        hurst_threshold: Hurst threshold for mean reversion regime (default 0.4)
        vix_prices: Optional VIX price series for volatility analysis
        current_vix: Current VIX value (required if vix_prices provided)
        vix_zscore_threshold: Z-score threshold for elevated VIX (default 2.0)

    Returns:
        MeanReversionAnalysis with test results and trade signal
    """
    if vr_lags is None:
        vr_lags = [2, 3, 5, 10, 21]

    # 1. Extract close prices
    closes = historical_prices[price_column].copy()

    # 2. Compute historical stretch series
    ma = closes.rolling(ma_period).mean()
    stretch = closes - ma
    stretch_clean = stretch.dropna()

    # 3. Compute current MA and current stretch
    current_ma = closes.tail(ma_period).mean()
    current_stretch = current_price - current_ma
    current_stretch_pct = (current_stretch / current_ma) * 100

    # 4. Run ADF test on historical stretch series
    adf_result = adfuller(stretch_clean, autolag="AIC")
    adf_statistic = adf_result[0]
    adf_pvalue = adf_result[1]
    adf_is_stationary = adf_pvalue < adf_pvalue_threshold

    # 5. Run KPSS test on historical stretch series
    kpss_result = kpss(stretch_clean, regression="c", nlags="auto")
    kpss_statistic = kpss_result[0]
    kpss_pvalue = kpss_result[1]
    kpss_is_stationary = kpss_pvalue > kpss_pvalue_threshold

    # Combine tests: mean reverting if ADF rejects unit root AND KPSS fails to reject stationarity
    confirmed_mean_reverting = adf_is_stationary and kpss_is_stationary

    stretch_test = StretchTestResult(
        adf_statistic=adf_statistic,
        adf_pvalue=adf_pvalue,
        adf_is_stationary=adf_is_stationary,
        kpss_statistic=kpss_statistic,
        kpss_pvalue=kpss_pvalue,
        kpss_is_stationary=kpss_is_stationary,
        confirmed_mean_reverting=confirmed_mean_reverting,
    )

    # 6. Compute Variance Ratios for each lag
    variance_ratios = []
    for lag in vr_lags:
        vr = variance_ratio(closes, lag)
        if vr < 0.95:
            interpretation = "mean_reverting"
        elif vr > 1.05:
            interpretation = "momentum"
        else:
            interpretation = "random_walk"
        variance_ratios.append(VarianceRatioResult(lag=lag, ratio=vr, interpretation=interpretation))

    # 7. Find optimal DTE: lag with lowest VR that is < 1
    mean_reverting_vrs = [(vr.lag, vr.ratio) for vr in variance_ratios if vr.ratio < 1]
    if mean_reverting_vrs:
        optimal_dte_lag = min(mean_reverting_vrs, key=lambda x: x[1])[0]
    else:
        optimal_dte_lag = None

    # 8. Compute Rolling VR using last rolling_window days
    # Use lag=5 as reference, or optimal lag if found
    reference_lag = optimal_dte_lag if optimal_dte_lag is not None else 5
    recent_closes = closes.tail(rolling_window + reference_lag)
    current_regime_vr = variance_ratio(recent_closes, reference_lag)

    # Determine regime from VR
    if current_regime_vr > 1:
        regime = "MOMENTUM"
    else:
        regime = "MEAN_REVERTING"

    # === NEW: Enhanced Analysis ===

    # 9. Calculate Z-Score of current stretch
    zscore_result = calculate_zscore(stretch_clean, current_stretch, ma_period)

    # 10. Calculate Half-Life from Ornstein-Uhlenbeck
    half_life_result = calculate_half_life(stretch_clean)

    # 11. Calculate Hurst Exponent
    hurst_result = calculate_hurst_exponent(closes.tail(hurst_lookback))

    # 12. VIX Analysis (if data provided)
    vix_analysis = None
    if vix_prices is not None and current_vix is not None:
        vix_analysis = analyze_vix(
            vix_prices,
            current_vix,
            ma_period,
            vix_zscore_threshold,
            adf_pvalue_threshold,
            kpss_pvalue_threshold,
        )

    # === Generate Enhanced Signal ===
    signal_reasons = []

    # Hard Filters (Safety - DO NOT REMOVE)
    # These protect against trending/crashing markets where mean reversion fails
    hurst_ok = hurst_result.hurst_exponent < hurst_threshold
    regime_ok = current_regime_vr <= 1.0

    # Soft Filters (Entry Quality)
    zscore_ok = zscore_result.current_zscore < zscore_threshold
    stationarity_ok = confirmed_mean_reverting

    # VIX Bonus (Conviction Multiplier, not a gatekeeper)
    vix_is_bonus = vix_analysis is not None and vix_analysis.vix_is_elevated

    # Determine final signal
    # Priority 1: Hard filters (RED_FLAG - do not trade)
    if not hurst_ok:
        signal = "RED_FLAG"
        signal_reasons.append(f"Hurst={hurst_result.hurst_exponent:.3f} >= {hurst_threshold} ({hurst_result.regime})")
        signal_reasons.append("Market is trending - mean reversion will fail")
    elif not regime_ok:
        signal = "RED_FLAG"
        signal_reasons.append(f"Momentum regime (VR={current_regime_vr:.3f} > 1.0)")
        signal_reasons.append("Drops may continue - do not catch falling knife")

    # Priority 2: Entry quality (WAIT - conditions not ideal)
    elif not zscore_ok:
        signal = "WAIT"
        signal_reasons.append(f"Z-Score={zscore_result.current_zscore:.2f} > {zscore_threshold}")
        signal_reasons.append("No significant dip - rubber band not stretched")
        if vix_analysis:
            signal_reasons.append(f"VIX Z-Score: {vix_analysis.vix_zscore:.2f} (for reference)")

    # Priority 3: All clear - GREEN_LIGHT with conviction level
    else:
        # Determine conviction level
        conviction_factors = []
        if vix_is_bonus:
            conviction_factors.append("VIX spike")
        if zscore_result.current_zscore < -2.0:
            conviction_factors.append("deep dip")
        if stationarity_ok:
            conviction_factors.append("stationarity confirmed")
        if half_life_result.is_valid and half_life_result.half_life_days < 10:
            conviction_factors.append("fast reversion")

        if len(conviction_factors) >= 3:
            signal = "GREEN_LIGHT (A+ SETUP)"
        elif len(conviction_factors) >= 1:
            signal = "GREEN_LIGHT (HIGH CONVICTION)"
        else:
            signal = "GREEN_LIGHT"

        signal_reasons.append(f"Hurst={hurst_result.hurst_exponent:.3f} - mean-reverting regime")
        signal_reasons.append(f"Z-Score={zscore_result.current_zscore:.2f} - dip detected")
        signal_reasons.append(f"Half-life={half_life_result.half_life_days:.1f}d -> DTE: {half_life_result.recommended_min_dte}-{half_life_result.recommended_max_dte}d")

        if conviction_factors:
            signal_reasons.append(f"Conviction: {', '.join(conviction_factors)}")

        if vix_analysis:
            vix_status = "RICH premiums" if vix_is_bonus else "normal premiums"
            signal_reasons.append(f"VIX={vix_analysis.current_vix:.1f} (Z={vix_analysis.vix_zscore:.2f}) - {vix_status}")

    return MeanReversionAnalysis(
        stretch_test=stretch_test,
        variance_ratios=variance_ratios,
        optimal_dte_lag=optimal_dte_lag,
        current_regime_vr=current_regime_vr,
        regime=regime,
        current_stretch=current_stretch,
        current_stretch_pct=current_stretch_pct,
        zscore=zscore_result,
        half_life=half_life_result,
        hurst=hurst_result,
        vix_analysis=vix_analysis,
        signal=signal,
        signal_reasons=signal_reasons,
    )


def generate_mean_reverting_series(n: int = 200, rho: float = 0.7, seed: int = 42) -> pd.Series:
    """Generate a mean-reverting AR(1) price series."""
    np.random.seed(seed)
    base_price = 5000
    noise = np.random.normal(0, 20, n)
    stretch = np.zeros(n)
    for i in range(1, n):
        stretch[i] = rho * stretch[i - 1] + noise[i]
    prices = base_price + stretch
    return pd.Series(prices)


def generate_random_walk_series(n: int = 200, seed: int = 42) -> pd.Series:
    """Generate a random walk price series."""
    np.random.seed(seed)
    base_price = 5000
    returns = np.random.normal(0, 20, n)
    prices = base_price + np.cumsum(returns)
    return pd.Series(prices)


def generate_trending_series(n: int = 200, trend: float = 2.0, seed: int = 42) -> pd.Series:
    """Generate a trending price series with momentum."""
    np.random.seed(seed)
    base_price = 5000
    noise = np.random.normal(0, 10, n)
    trend_component = np.arange(n) * trend
    prices = base_price + trend_component + np.cumsum(noise * 0.5)
    return pd.Series(prices)


def generate_vix_series(n: int = 200, seed: int = 42) -> pd.Series:
    """Generate a VIX-like mean-reverting series."""
    np.random.seed(seed)
    base_vix = 18
    vix = np.zeros(n)
    vix[0] = base_vix
    for i in range(1, n):
        # Strong mean reversion with occasional spikes
        shock = np.random.normal(0, 2) + (0.1 if np.random.random() > 0.95 else 0) * 10
        vix[i] = vix[i - 1] + 0.1 * (base_vix - vix[i - 1]) + shock
        vix[i] = max(10, vix[i])  # VIX floor
    return pd.Series(vix)


def test_with_synthetic_data(verbose: bool = False):
    """Test the analysis with synthetic mean-reverting, random walk, and trending data."""
    print("Running unit tests...", end=" ")

    # Test 1: Mean-reverting series with significant dip should return GREEN_LIGHT
    mr_prices = generate_mean_reverting_series(n=200, rho=0.7)
    mr_df = pd.DataFrame({"Close": mr_prices})
    stretch_std = (mr_prices - mr_prices.rolling(20).mean()).std()
    current_price = mr_prices.iloc[-1] - 3 * stretch_std
    result = analyze_mean_reversion(mr_df, current_price)
    test1_pass = "GREEN_LIGHT" in result.signal

    # Test 2: Random walk should return RED_FLAG
    rw_prices = generate_random_walk_series(n=200)
    rw_df = pd.DataFrame({"Close": rw_prices})
    result = analyze_mean_reversion(rw_df, rw_prices.iloc[-1] - 50)
    test2_pass = result.signal == "RED_FLAG" or result.signal == "WAIT"

    # Test 3: Trending series should return RED_FLAG
    tr_prices = generate_trending_series(n=200, trend=3.0)
    tr_df = pd.DataFrame({"Close": tr_prices})
    result = analyze_mean_reversion(tr_df, tr_prices.iloc[-1] - 50)
    test3_pass = result.signal == "RED_FLAG" or result.signal == "WAIT"

    # Test 4: Mean-reverting with no dip should return WAIT
    mr_prices = generate_mean_reverting_series(n=200, rho=0.7, seed=99)
    mr_df = pd.DataFrame({"Close": mr_prices})
    result = analyze_mean_reversion(mr_df, mr_prices.iloc[-1])  # No dip
    test4_pass = result.signal == "WAIT" or "GREEN_LIGHT" not in result.signal

    all_passed = test1_pass and test2_pass and test3_pass and test4_pass
    print(f"{'All 4 tests passed.' if all_passed else 'SOME TESTS FAILED!'}")

    if verbose or not all_passed:
        print(f"  Test 1 (mean-reverting + dip → GREEN): {'PASS' if test1_pass else 'FAIL'}")
        print(f"  Test 2 (random walk → not GREEN): {'PASS' if test2_pass else 'FAIL'}")
        print(f"  Test 3 (trending → not GREEN): {'PASS' if test3_pass else 'FAIL'}")
        print(f"  Test 4 (no dip → WAIT): {'PASS' if test4_pass else 'FAIL'}")

    return all_passed


def demo_spx_analysis(show_plot: bool = False, save_plot: str | None = None):
    """Demo with sample SPX-like price data including VIX analysis."""
    # Generate SPX-like price data with mean-reverting characteristics
    np.random.seed(123)
    n = 150
    base_price = 5800
    trend = np.linspace(0, 100, n)
    mean_reversion = np.zeros(n)
    for i in range(1, n):
        mean_reversion[i] = 0.75 * mean_reversion[i - 1] + np.random.normal(0, 20)
    prices = base_price + trend + mean_reversion

    dates = pd.date_range(end=datetime.now(), periods=n, freq="D")
    df = pd.DataFrame(
        {
            "Date": dates,
            "Open": prices - np.random.uniform(0, 5, n),
            "High": prices + np.random.uniform(5, 15, n),
            "Low": prices - np.random.uniform(5, 15, n),
            "Close": prices,
        }
    )

    vix_prices = generate_vix_series(n=n, seed=456)
    current_vix = vix_prices.iloc[-1] + 8

    stretch_std = (pd.Series(prices) - pd.Series(prices).rolling(20).mean()).std()
    current_price = prices[-1] - 2.5 * stretch_std

    print(f"\nSynthetic data: {len(df)} days | SPX: {current_price:.2f} | VIX: {current_vix:.2f}")

    result = analyze_mean_reversion(
        df,
        current_price,
        vix_prices=vix_prices,
        current_vix=current_vix,
    )

    print_analysis_results(result, "SPX Mean Reversion Analysis (Synthetic)")

    if show_plot or save_plot:
        plot_analysis(
            df,
            result,
            current_price,
            vix_prices=vix_prices,
            current_vix=current_vix,
            save_path=save_plot,
            show=show_plot,
        )


def print_analysis_results(result: MeanReversionAnalysis, title: str = "SPX Mean Reversion Analysis"):
    """Print formatted analysis results."""
    print(f"\n{'=' * 70}")
    print(f" {title}")
    print(f"{'=' * 70}")

    # Signal first (most important)
    if "GREEN_LIGHT" in result.signal:
        signal_color = "\033[92m"  # Green
    elif result.signal == "WAIT":
        signal_color = "\033[93m"  # Yellow
    else:
        signal_color = "\033[91m"  # Red
    reset = "\033[0m"
    print(f"\n SIGNAL: {signal_color}{result.signal}{reset}")
    for reason in result.signal_reasons:
        print(f"   - {reason}")

    # Safety Filters (Hard)
    print(f"\n REGIME FILTERS (hard):")
    hurst_ok = result.hurst.hurst_exponent < 0.5
    vr_ok = result.current_regime_vr <= 1.0
    print(f"   [{'X' if hurst_ok else ' '}] Hurst < 0.5 ({result.hurst.hurst_exponent:.3f} - {result.hurst.regime})")
    print(f"   [{'X' if vr_ok else ' '}] VR <= 1.0 ({result.current_regime_vr:.3f})")

    # Entry Quality (Soft)
    print(f"\n ENTRY QUALITY (soft):")
    zscore_ok = result.zscore.current_zscore < -1.0
    print(f"   [{'X' if zscore_ok else ' '}] Z-Score < -1.0 ({result.zscore.current_zscore:.2f})")
    print(f"   [{'X' if result.stretch_test.confirmed_mean_reverting else ' '}] Stationarity confirmed")

    # Conviction Boosters
    print(f"\n CONVICTION BOOSTERS:")
    deep_dip = result.zscore.current_zscore < -2.0
    print(f"   [{'X' if deep_dip else ' '}] Deep dip (Z < -2.0)")
    if result.vix_analysis:
        print(f"   [{'X' if result.vix_analysis.vix_is_elevated else ' '}] VIX elevated (rich premiums)")
    fast_reversion = result.half_life.is_valid and result.half_life.half_life_days < 10
    print(f"   [{'X' if fast_reversion else ' '}] Fast reversion (HL < 10d)")

    # Key metrics
    print(f"\n KEY METRICS:")
    print(f"   Current Stretch:  {result.current_stretch:+.2f} pts ({result.current_stretch_pct:+.2f}%)")
    print(f"   Z-Score:          {result.zscore.current_zscore:.2f} (threshold: {result.zscore.zscore_threshold})")
    print(f"   Hurst Exponent:   {result.hurst.hurst_exponent:.3f} ({result.hurst.regime})")
    print(f"   Half-Life:        {result.half_life.half_life_days:.1f} days")
    print(f"   Recommended DTE:  {result.half_life.recommended_min_dte}-{result.half_life.recommended_max_dte} days")

    # VIX info if available
    if result.vix_analysis:
        print(f"\n VIX:")
        print(f"   Current:    {result.vix_analysis.current_vix:.2f} (MA: {result.vix_analysis.vix_ma:.2f})")
        print(f"   Z-Score:    {result.vix_analysis.vix_zscore:.2f} ({'ELEVATED' if result.vix_analysis.vix_is_elevated else 'normal'})")

    # Stationarity tests
    print(f"\n STATIONARITY TESTS:")
    adf_status = "stationary" if result.stretch_test.adf_is_stationary else "non-stationary"
    kpss_status = "stationary" if result.stretch_test.kpss_is_stationary else "non-stationary"
    print(f"   ADF:  {result.stretch_test.adf_statistic:.3f} (p={result.stretch_test.adf_pvalue:.4f}) -> {adf_status}")
    print(f"   KPSS: {result.stretch_test.kpss_statistic:.3f} (p={result.stretch_test.kpss_pvalue:.4f}) -> {kpss_status}")

    # Variance ratios
    print(f"\n VARIANCE RATIOS:")
    vr_str = "   " + " | ".join([f"Lag {vr.lag}: {vr.ratio:.3f}" for vr in result.variance_ratios])
    print(vr_str)

    print(f"\n{'=' * 70}\n")


def plot_analysis(
    historical_prices: pd.DataFrame,
    result: MeanReversionAnalysis,
    current_price: float,
    price_column: str = "Close",
    ma_period: int = 20,
    vix_prices: pd.Series | None = None,
    current_vix: float | None = None,
    save_path: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot mean reversion analysis with stretch, half-life bands, and signal zones.

    Args:
        historical_prices: DataFrame with OHLC data
        result: MeanReversionAnalysis result from analyze_mean_reversion()
        current_price: Current price used in analysis
        price_column: Column name for prices
        ma_period: Moving average period
        vix_prices: Optional VIX series for subplot
        current_vix: Current VIX value
        save_path: Path to save figure (optional)
        show: Whether to display the plot

    Returns:
        matplotlib Figure object
    """
    closes = historical_prices[price_column]
    dates = historical_prices.get("Date", pd.RangeIndex(len(closes)))

    # Calculate stretch series
    ma = closes.rolling(ma_period).mean()
    stretch = closes - ma
    stretch_std = stretch.rolling(ma_period).std()

    # Determine subplot layout
    n_plots = 3 if vix_prices is not None else 2
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 4 * n_plots), sharex=True)

    # Color scheme
    green = "#22c55e"
    yellow = "#eab308"
    red = "#ef4444"
    blue = "#3b82f6"
    gray = "#6b7280"

    if "GREEN_LIGHT" in result.signal:
        signal_color = green
    elif result.signal == "WAIT":
        signal_color = yellow
    else:
        signal_color = red

    # === Plot 1: Price with MA and current position ===
    ax1 = axes[0]
    ax1.plot(dates, closes, color=blue, linewidth=1.5, label="SPX Close")
    ax1.plot(dates, ma, color=gray, linewidth=1, linestyle="--", label=f"MA({ma_period})")

    # Current price marker
    ax1.axhline(current_price, color=signal_color, linestyle=":", linewidth=1.5, alpha=0.8)
    ax1.scatter([dates.iloc[-1]], [current_price], color=signal_color, s=100, zorder=5, marker="o")
    ax1.annotate(
        f"Current: {current_price:.0f}",
        xy=(dates.iloc[-1], current_price),
        xytext=(10, 0),
        textcoords="offset points",
        fontsize=9,
        color=signal_color,
        fontweight="bold",
    )

    ax1.set_ylabel("Price", fontsize=10)
    ax1.set_title(f"SPX Price | Signal: {result.signal}", fontsize=12, fontweight="bold", color=signal_color)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # === Plot 2: Stretch (Z-Score) with bands ===
    ax2 = axes[1]

    # Z-score series
    zscore_series = stretch / stretch_std

    ax2.fill_between(dates, -2, 2, color=gray, alpha=0.1, label="Normal zone")
    ax2.fill_between(dates, -3, -2, color=green, alpha=0.2, label="Buy zone (Z < -2)")
    ax2.fill_between(dates, 2, 3, color=red, alpha=0.2, label="Overbought (Z > 2)")

    ax2.plot(dates, zscore_series, color=blue, linewidth=1.5)
    ax2.axhline(0, color=gray, linewidth=0.5)
    ax2.axhline(-2, color=green, linewidth=1, linestyle="--", alpha=0.7)
    ax2.axhline(2, color=red, linewidth=1, linestyle="--", alpha=0.7)

    # Current Z-score marker
    ax2.scatter([dates.iloc[-1]], [result.zscore.current_zscore], color=signal_color, s=100, zorder=5)
    ax2.annotate(
        f"Z = {result.zscore.current_zscore:.2f}",
        xy=(dates.iloc[-1], result.zscore.current_zscore),
        xytext=(10, 0),
        textcoords="offset points",
        fontsize=9,
        color=signal_color,
        fontweight="bold",
    )

    ax2.set_ylabel("Z-Score", fontsize=10)
    ax2.set_title(
        f"Stretch Z-Score | Half-Life: {result.half_life.half_life_days:.1f}d | "
        f"Hurst: {result.hurst.hurst_exponent:.2f} ({result.hurst.regime})",
        fontsize=11,
    )
    ax2.legend(loc="upper left", fontsize=8)
    ax2.set_ylim(-4, 4)
    ax2.grid(True, alpha=0.3)

    # === Plot 3: VIX (if available) ===
    if vix_prices is not None and len(axes) > 2:
        ax3 = axes[2]
        vix_ma = vix_prices.rolling(ma_period).mean()
        vix_dates = dates.iloc[-len(vix_prices) :]

        ax3.plot(vix_dates, vix_prices.values, color="#f59e0b", linewidth=1.5, label="VIX")
        ax3.plot(vix_dates, vix_ma.values, color=gray, linewidth=1, linestyle="--", label=f"VIX MA({ma_period})")

        # VIX spike zone
        ax3.axhline(20, color=gray, linewidth=0.5, linestyle=":")
        ax3.axhline(30, color=red, linewidth=1, linestyle="--", alpha=0.5)

        if current_vix is not None:
            vix_color = green if result.vix_analysis and result.vix_analysis.vix_is_elevated else gray
            ax3.scatter([vix_dates.iloc[-1]], [current_vix], color=vix_color, s=100, zorder=5)
            ax3.annotate(
                f"VIX: {current_vix:.1f}",
                xy=(vix_dates.iloc[-1], current_vix),
                xytext=(10, 0),
                textcoords="offset points",
                fontsize=9,
                color=vix_color,
                fontweight="bold",
            )

        vix_zscore = result.vix_analysis.vix_zscore if result.vix_analysis else 0
        elevated = "ELEVATED" if result.vix_analysis and result.vix_analysis.vix_is_elevated else "normal"
        ax3.set_ylabel("VIX", fontsize=10)
        ax3.set_title(f"VIX | Z-Score: {vix_zscore:.2f} ({elevated})", fontsize=11)
        ax3.legend(loc="upper left", fontsize=9)
        ax3.grid(True, alpha=0.3)

    # Format x-axis
    axes[-1].set_xlabel("Date", fontsize=10)

    # Add summary box
    summary_text = (
        f"Signal: {result.signal}\n"
        f"Z-Score: {result.zscore.current_zscore:.2f}\n"
        f"Hurst: {result.hurst.hurst_exponent:.3f}\n"
        f"Half-Life: {result.half_life.half_life_days:.1f}d\n"
        f"Rec. DTE: {result.half_life.recommended_min_dte}-{result.half_life.recommended_max_dte}d"
    )
    props = dict(boxstyle="round,pad=0.5", facecolor=signal_color, alpha=0.15, edgecolor=signal_color)
    ax1.text(
        0.02,
        0.98,
        summary_text,
        transform=ax1.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=props,
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    if show:
        plt.show()

    return fig


def demo_live_spx_analysis(show_plot: bool = False, save_plot: str | None = None):
    """
    Demo with live SPX and VIX data from Financial Modeling Prep.

    Requires FMP_API_KEY environment variable to be set.
    Get a free API key at: https://financialmodelingprep.com/developer/docs/
    """
    try:
        api_key = get_fmp_api_key()
    except FMPError as e:
        print(f"\n{e}")
        print("\nSkipping live demo. Set FMP_API_KEY to enable.")
        return

    print("Fetching live data from Financial Modeling Prep...", end=" ", flush=True)

    try:
        # Fetch SPX and VIX data
        spx_df, vix_series = fetch_spx_and_vix(days=150, api_key=api_key)
        current_spx = get_current_price("^GSPC", api_key)
        current_vix = get_current_price("^VIX", api_key)
        print("done.")

        print(f"\nData: {len(spx_df)} days | SPX: {current_spx:.2f} | VIX: {current_vix:.2f}")

        # Run analysis
        result = analyze_mean_reversion(
            spx_df,
            current_spx,
            vix_prices=vix_series,
            current_vix=current_vix,
        )

        print_analysis_results(result, "SPX Mean Reversion Analysis (Live)")

        if show_plot or save_plot:
            plot_analysis(
                spx_df,
                result,
                current_spx,
                vix_prices=vix_series,
                current_vix=current_vix,
                save_path=save_plot,
                show=show_plot,
            )

    except httpx.HTTPStatusError as e:
        print(f"\nHTTP Error: {e}")
        print("Check your API key and network connection.")
    except FMPError as e:
        print(f"\nFMP Error: {e}")


def main():
    """Run tests and demo."""
    import sys

    # Check for command-line flags
    run_tests = "--test" in sys.argv or "-t" in sys.argv
    run_synthetic = "--synthetic" in sys.argv or "-s" in sys.argv
    show_plot = "--plot" in sys.argv or "-p" in sys.argv
    save_plot = None
    for arg in sys.argv:
        if arg.startswith("--save="):
            save_plot = arg.split("=", 1)[1]

    # If live API key is available, default to live demo
    has_api_key = bool(os.environ.get("FMP_API_KEY"))

    if run_tests or (not has_api_key and not run_synthetic):
        test_with_synthetic_data()

    if run_synthetic or not has_api_key:
        if not has_api_key:
            demo_spx_analysis(show_plot=show_plot, save_plot=save_plot)
            print("\n" + "-" * 70)
            print("NOTE: Using synthetic data. For live SPX/VIX analysis:")
            print("  1. Get a free API key at https://financialmodelingprep.com/developer/docs/")
            print("  2. Run: FMP_API_KEY=your_key uv run python main.py")
        else:
            demo_spx_analysis(show_plot=show_plot, save_plot=save_plot)
    else:
        # Live demo with real data
        demo_live_spx_analysis(show_plot=show_plot, save_plot=save_plot)


if __name__ == "__main__":
    main()
