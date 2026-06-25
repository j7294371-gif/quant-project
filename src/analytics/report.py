from decimal import Decimal
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box


console = Console()


def print_backtest_report(metrics, trades, warnings: list[str] | None = None):
    """Print backtest results as Rich tables."""
    console.print()

    # Main metrics table
    table = Table(title="Backtest Results", box=box.ROUNDED, title_style="bold cyan")
    table.add_column("Metric", style="dim", width=25)
    table.add_column("Value", justify="right")

    total_ret = float(metrics.total_return) * 100
    color = "green" if total_ret >= 0 else "red"
    table.add_row("Total Return", f"[{color}]{total_ret:+.2f}%[/{color}]")
    table.add_row("Annualized Return", f"[{color}]{float(metrics.annualized_return)*100:+.2f}%[/{color}]")
    table.add_row("Benchmark (B&H)", f"{float(metrics.benchmark_return)*100:+.2f}%")
    table.add_row("Sharpe Ratio", f"{metrics.sharpe_ratio:.3f}")
    table.add_row("Sortino Ratio", f"{metrics.sortino_ratio:.3f}")

    calmar_str = f"{metrics.calmar_ratio:.3f}" if metrics.calmar_ratio != float('inf') else "∞"
    table.add_row("Calmar Ratio", calmar_str)
    table.add_row("Max Drawdown", f"[red]{float(metrics.max_drawdown)*100:.2f}%[/red]")
    table.add_row("Win Rate", f"{metrics.win_rate*100:.1f}%")

    pf_str = f"{metrics.profit_factor:.3f}" if metrics.profit_factor != float('inf') else "∞"
    table.add_row("Profit Factor", pf_str)
    table.add_row("Total Trades", str(metrics.total_trades))
    table.add_row("Avg Trade Duration", f"{metrics.avg_trade_duration_hours:.1f}h")

    console.print(table)

    # Trade details (last 20)
    if trades:
        console.print()
        trades_table = Table(title="Recent Trades (last 20)", box=box.SIMPLE)
        trades_table.add_column("Time", style="dim", width=12)
        trades_table.add_column("Side", width=6)
        trades_table.add_column("Price", justify="right", width=12)
        trades_table.add_column("Qty", justify="right", width=10)
        trades_table.add_column("Fee", justify="right", width=10)
        trades_table.add_column("Reason", width=30)

        from datetime import datetime, timezone
        for trade in trades[-20:]:
            dt = datetime.fromtimestamp(trade.timestamp / 1000, tz=timezone.utc)
            time_str = dt.strftime("%m-%d %H:%M")
            side_color = "green" if trade.side == "buy" else "red"
            price_str = f"${float(trade.filled_price):.2f}" if trade.filled_price else "N/A"
            trades_table.add_row(
                time_str,
                f"[{side_color}]{trade.side.upper()}[/{side_color}]",
                price_str,
                f"{float(trade.filled_quantity):.6f}",
                f"${float(trade.fee):.4f}",
                trade.reason[:28],
            )

        console.print(trades_table)

    # Warnings panel
    if warnings:
        console.print()
        warning_text = "\n".join(f"⚠ {w}" for w in warnings)
        console.print(Panel(warning_text, title="Warnings", border_style="yellow"))

    console.print()


def print_session_summary(
    mode: str,
    start_time: float,
    trades: list,
    realized_pnl: Decimal,
    unrealized_pnl: Decimal,
    circuit_events: int,
    warnings: list[str],
):
    """Print paper/live session summary on exit."""
    import time
    from datetime import datetime, timezone

    duration_s = time.monotonic() - start_time
    hours = int(duration_s // 3600)
    minutes = int((duration_s % 3600) // 60)

    buy_count = sum(1 for t in trades if t.side == "buy")
    sell_count = sum(1 for t in trades if t.side == "sell")

    start_dt = datetime.fromtimestamp(start_time, tz=timezone.utc)

    pnl_color = "green" if realized_pnl >= 0 else "red"
    unrealized_color = "green" if unrealized_pnl >= 0 else "red"

    lines = [
        f"Duration:          {hours}h {minutes}m",
        f"Total Trades:      {len(trades)} ({buy_count} buy / {sell_count} sell)",
        f"Realized PnL:      [{pnl_color}]+${float(realized_pnl):.2f} USDT[/{pnl_color}]",
        f"Unrealized PnL:    [{unrealized_color}]${float(unrealized_pnl):.2f} USDT[/{unrealized_color}]",
        f"Circuit Events:    {circuit_events}",
    ]

    title = f"Session Summary — {mode.upper()} Mode ({start_dt.strftime('%Y-%m-%d %H:%M UTC')})"
    border = "green" if realized_pnl >= 0 else "red"
    console.print()
    console.print(Panel("\n".join(lines), title=title, border_style=border))
    console.print()


def generate_html_report(metrics, trades, equity_curve, output_path: str):
    """Generate standalone HTML report with Plotly charts."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Equity curve
    eq_values = [float(e) for e in equity_curve]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.6, 0.4],
        subplot_titles=("Equity Curve", "Drawdown"),
    )

    fig.add_trace(
        go.Scatter(y=eq_values, mode="lines", name="Equity", line=dict(color="blue")),
        row=1, col=1,
    )

    # Buy/sell markers
    buy_times = []
    buy_prices = []
    sell_times = []
    sell_prices = []

    for t in trades:
        if t.side == "buy" and t.filled_price:
            buy_times.append(len(buy_prices))  # Approximate index
            buy_prices.append(float(t.filled_price))
        elif t.side == "sell" and t.filled_price:
            sell_times.append(len(sell_prices))
            sell_prices.append(float(t.filled_price))

    if buy_prices:
        fig.add_trace(
            go.Scatter(y=buy_prices, mode="markers", name="Buy",
                       marker=dict(color="green", symbol="triangle-up", size=10)),
            row=1, col=1,
        )
    if sell_prices:
        fig.add_trace(
            go.Scatter(y=sell_prices, mode="markers", name="Sell",
                       marker=dict(color="red", symbol="triangle-down", size=10)),
            row=1, col=1,
        )

    # Drawdown
    peak = eq_values[0]
    drawdowns = []
    for v in eq_values:
        if v > peak:
            peak = v
        dd = (peak - v) / peak * 100 if peak > 0 else 0
        drawdowns.append(-dd)

    fig.add_trace(
        go.Scatter(y=drawdowns, mode="lines", name="Drawdown %",
                   fill="tozeroy", line=dict(color="red")),
        row=2, col=1,
    )

    fig.update_layout(height=800, showlegend=True,
                      title_text="Quant Project — Backtest Report")
    fig.write_html(output_path)
    console.print(f"[green]HTML report saved to {output_path}[/green]")


def print_health_check(
    start_time: float,
    mode: str,
    symbols: list[str],
    total_trades: int,
    daily_pnl: Decimal,
    circuit_breaker,
    ws_connected: bool,
    last_bar_time: int,
):
    """Print hourly health check Rich Panel."""
    import time
    from datetime import datetime, timezone

    duration_s = time.monotonic() - start_time
    hours = int(duration_s // 3600)
    minutes = int((duration_s % 3600) // 60)

    circuit_status = "TRIPPED" if circuit_breaker.is_tripped else "NORMAL"

    last_bar_str = "N/A"
    if last_bar_time > 0:
        last_bar_str = datetime.fromtimestamp(
            last_bar_time / 1000, tz=timezone.utc
        ).strftime("%Y-%m-%d %H:%M UTC")

    ws_status = "Connected" if ws_connected else "Disconnected"

    lines = [
        f"Uptime:        {hours}h {minutes}m",
        f"Mode:          {mode.upper()}",
        f"Active Symbols: {', '.join(symbols)}",
        f"Total Trades:  {total_trades}",
        f"Daily PnL:     ${float(daily_pnl):+.2f}",
        f"Circuit:       {circuit_status}",
        f"WS Status:     {ws_status}",
        f"Last Bar:      {last_bar_str}",
    ]

    # Determine color
    now_ms = int(time.time() * 1000)
    bar_stale = (now_ms - last_bar_time) > 5 * 60 * 1000 if last_bar_time > 0 else True

    if circuit_breaker.is_tripped or bar_stale:
        border = "red"
    elif not ws_connected or bar_stale:
        border = "yellow"
    else:
        border = "green"

    now_dt = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    title = f"System Health Check ({now_dt})"
    console.print()
    console.print(Panel("\n".join(lines), title=title, border_style=border))
