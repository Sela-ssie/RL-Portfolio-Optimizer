import numpy as np

TRADING_DAYS = 252


def value_path_to_log_returns(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return np.array([], dtype=np.float64)
    return np.diff(np.log(values))


def annualized_return(values: np.ndarray, periods_per_year: int = TRADING_DAYS) -> float:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2 or values[0] <= 0:
        return np.nan
    total_years = (len(values) - 1) / periods_per_year
    if total_years <= 0:
        return np.nan
    return (values[-1] / values[0]) ** (1.0 / total_years) - 1.0


def annualized_vol(log_returns: np.ndarray, periods_per_year: int = TRADING_DAYS) -> float:
    log_returns = np.asarray(log_returns, dtype=np.float64)
    if len(log_returns) < 2:
        return np.nan
    return log_returns.std(ddof=1) * np.sqrt(periods_per_year)


def sharpe_ratio(
    log_returns: np.ndarray,
    risk_free_rate_annual: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    log_returns = np.asarray(log_returns, dtype=np.float64)
    if len(log_returns) < 2:
        return np.nan

    # Convert annual RF to per-period (approx for small rates)
    rf_per_period = risk_free_rate_annual / periods_per_year
    excess = log_returns - rf_per_period

    denom = excess.std(ddof=1)
    if denom < 1e-12:
        return np.nan
    return excess.mean() / denom * np.sqrt(periods_per_year)


def max_drawdown(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return np.nan
    peak = np.maximum.accumulate(values)
    dd = 1.0 - (values / (peak + 1e-12))   # drawdown as positive fraction
    return float(dd.max())

def sortino_ratio(log_returns: np.ndarray, periods_per_year: int = TRADING_DAYS) -> float:
    log_returns = np.asarray(log_returns, dtype=np.float64)
    if len(log_returns) < 2:
        return np.nan
    downside = log_returns[log_returns < 0]
    if len(downside) < 2:
        return np.nan
    denom = downside.std(ddof=1)
    if denom < 1e-12:
        return np.nan
    return log_returns.mean() / denom * np.sqrt(periods_per_year)

def summary_metrics(values: np.ndarray, turnover: np.ndarray | None = None) -> dict:
    lr = value_path_to_log_returns(values)
    out = {
        "final": float(values[-1]) if len(values) else np.nan,
        "ann_return": annualized_return(values),
        "ann_vol": annualized_vol(lr),
        "sharpe": sharpe_ratio(lr),
        "max_dd": max_drawdown(values),
    }
    if turnover is not None and len(turnover) > 0:
        out["avg_turnover"] = float(np.mean(turnover))
        out["max_turnover"] = float(np.max(turnover))
    else:
        out["avg_turnover"] = 0.0
        out["max_turnover"] = 0.0
    out["sortino"] = sortino_ratio(lr)
    out["calmar"] = out["ann_return"] / (out["max_dd"] + 1e-12) if (np.isfinite(out["ann_return"]) and np.isfinite(out["max_dd"])) else np.nan
    return out


def print_metrics_table(results: dict[str, dict]):
    # Simple aligned printing (no external deps)
    headers = ["Strategy", "Final", "AnnRet", "AnnVol", "Sharpe", "MaxDD", "AvgTO(gross)", "Sortino", "Calmar"]
    colw = [max(len(headers[0]), 18), 10, 10, 10, 10, 10, 14, 10, 10]

    def fmt(x):
        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
            return "nan"
        return f"{x:.4f}"

    line = (
        f"{headers[0]:<{colw[0]}}"
        f"{headers[1]:>{colw[1]}}"
        f"{headers[2]:>{colw[2]}}"
        f"{headers[3]:>{colw[3]}}"
        f"{headers[4]:>{colw[4]}}"
        f"{headers[5]:>{colw[5]}}"
        f"{headers[6]:>{colw[6]}}"
        f"{headers[7]:>{colw[7]}}"
        f"{headers[8]:>{colw[8]}}"
    )
    print(line)
    print("-" * sum(colw))

    for name, m in results.items():
        print(
            f"{name:<{colw[0]}}"
            f"{fmt(m['final']):>{colw[1]}}"
            f"{fmt(m['ann_return']):>{colw[2]}}"
            f"{fmt(m['ann_vol']):>{colw[3]}}"
            f"{fmt(m['sharpe']):>{colw[4]}}"
            f"{fmt(m['max_dd']):>{colw[5]}}"
            f"{fmt(m['avg_turnover']):>{colw[6]}}"
            f"{fmt(m.get('sortino', np.nan)):>{colw[7]}}"
            f"{fmt(m.get('calmar', np.nan)):>{colw[8]}}"
        )