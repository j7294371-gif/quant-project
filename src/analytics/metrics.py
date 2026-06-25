from dataclasses import dataclass
from decimal import Decimal
import numpy as np
from loguru import logger


@dataclass(frozen=True)
class Metrics:
    total_return: Decimal
    annualized_return: Decimal
    benchmark_return: Decimal
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float  # float('inf') if MDD=0
    max_drawdown: Decimal
    win_rate: float
    profit_factor: float  # float('inf') if total_losses=0
    total_trades: int
    avg_trade_duration_hours: float


def _timeframe_to_periods_per_year(timeframe: str) -> int:
    """Convert timeframe to periods per year. Crypto: 8760 hours/year."""
    unit = timeframe[-1]
    value = int(timeframe[:-1])
    if unit == "m":
        return 8760 * 60 // value  # minutes
    elif unit == "h":
        return 8760 // value  # hours
    elif unit == "d":
        return 365 // value  # days
    else:
        raise ValueError(f"Unknown timeframe: {timeframe}")


def calculate_metrics(
    result,
    timeframe: str,
    benchmark_close_series: list[float] | None = None,
    risk_free_rate: float = 0.02,
) -> Metrics:
    """
    Calculate all metrics from BacktestResult.

    Steps: period returns -> annualization factor -> all indicators -> benchmark
    """
    equity_curve = result.equity_curve
    trades = result.trades

    if len(equity_curve) < 2:
        return Metrics(
            total_return=Decimal("0"),
            annualized_return=Decimal("0"),
            benchmark_return=Decimal("0"),
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            max_drawdown=Decimal("0"),
            win_rate=0.0,
            profit_factor=0.0,
            total_trades=0,
            avg_trade_duration_hours=0.0,
        )

    n_bars = len(equity_curve) - 1
    periods_per_year = _timeframe_to_periods_per_year(timeframe)

    # Step 1: Period returns (vectorized)
    equity_arr = np.array([float(e) for e in equity_curve])
    returns_arr = np.diff(equity_arr) / equity_arr[:-1]

    # Step 2: Total & annualized return
    initial = float(equity_curve[0])
    final = float(equity_curve[-1])
    total_return = Decimal(str((final - initial) / initial)) if initial > 0 else Decimal("0")

    if total_return > Decimal("-1") and n_bars > 0:
        annualized = (1.0 + float(total_return)) ** (periods_per_year / n_bars) - 1.0
    else:
        annualized = 0.0
    annualized_return = Decimal(str(annualized))

    # Step 3a: Sharpe ratio
    period_std = float(np.std(returns_arr, ddof=1))
    if period_std > 0:
        sharpe = (annualized - risk_free_rate) / (period_std * np.sqrt(periods_per_year))
    else:
        sharpe = 0.0

    # Step 3b: Sortino ratio
    downside = returns_arr[returns_arr < 0]
    if len(downside) > 0:
        down_std = float(np.std(downside, ddof=1))
        if down_std > 0:
            sortino = (annualized - risk_free_rate) / (down_std * np.sqrt(periods_per_year))
        else:
            sortino = 0.0
    else:
        sortino = float('inf') if annualized > risk_free_rate else 0.0

    # Step 3c: Max drawdown (vectorized)
    peak_arr = np.maximum.accumulate(equity_arr)
    drawdowns = (peak_arr - equity_arr) / peak_arr
    mdd = Decimal(str(float(np.max(drawdowns))))

    # Step 3d: Calmar ratio
    if float(mdd) > 0:
        calmar = annualized / float(mdd)
    else:
        calmar = float('inf')

    # Step 4: Trade statistics
    buy_trades = [t for t in trades if t.side == "buy" and t.status.value == "filled"]
    sell_trades = [t for t in trades if t.side == "sell" and t.status.value == "filled"]

    # Pair trades (simple: buy then next sell)
    paired = min(len(buy_trades), len(sell_trades))
    pnls = np.zeros(paired)
    durations_hours = np.zeros(paired)

    for i in range(paired):
        buy = buy_trades[i]
        sell = sell_trades[i]
        buy_val = float(buy.filled_quantity) * float(buy.filled_price) if buy.filled_price else 0
        sell_val = float(sell.filled_quantity) * float(sell.filled_price) if sell.filled_price else 0
        pnls[i] = sell_val - buy_val - float(buy.fee) - float(sell.fee)
        durations_hours[i] = (sell.timestamp - buy.timestamp) / 3600000.0

    winning_pnls = pnls[pnls > 0]
    losing_pnls = pnls[pnls < 0]
    total_profit = float(np.sum(winning_pnls))
    total_loss = abs(float(np.sum(losing_pnls)))

    winning = len(winning_pnls)
    losing = len(losing_pnls)
    total_trades = winning + losing
    win_rate = winning / total_trades if total_trades > 0 else 0.0
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
    avg_duration = float(np.mean(durations_hours)) if len(durations_hours) > 0 else 0.0

    # Step 5: Benchmark return
    benchmark_return = Decimal("0")
    if benchmark_close_series and len(benchmark_close_series) > 1:
        bench_ret = (benchmark_close_series[-1] - benchmark_close_series[0]) / benchmark_close_series[0]
        benchmark_return = Decimal(str(bench_ret))

    return Metrics(
        total_return=total_return,
        annualized_return=annualized_return,
        benchmark_return=benchmark_return,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        max_drawdown=mdd,
        win_rate=win_rate,
        profit_factor=profit_factor,
        total_trades=total_trades,
        avg_trade_duration_hours=avg_duration,
    )


def annualize_return(total_return: Decimal, n_bars: int, timeframe: str) -> Decimal:
    periods_per_year = _timeframe_to_periods_per_year(timeframe)
    if n_bars <= 0:
        return Decimal("0")
    ann = (1.0 + float(total_return)) ** (periods_per_year / n_bars) - 1.0
    return Decimal(str(ann))


def annualize_volatility(period_returns: np.ndarray, timeframe: str) -> float:
    periods_per_year = _timeframe_to_periods_per_year(timeframe)
    return float(np.std(period_returns, ddof=1)) * np.sqrt(periods_per_year)
