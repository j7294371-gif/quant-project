import time
from decimal import Decimal
from datetime import datetime, timezone
from src.orchestrator.base import BaseOrchestrator
from src.strategy.base import SignalAction
from src.execution.backtest import BacktestResult


class BacktestOrchestrator(BaseOrchestrator):
    def __init__(
        self,
        strategy,
        execution,
        symbols: list[str],
        risk_config,
        state_store,
        circuit_breaker,
        logger,
        data_df,
        initial_equity: Decimal = Decimal("10000"),
        is_oos: bool = False,
        train_result: BacktestResult | None = None,
        risk_manager=None,
    ):
        super().__init__(strategy, execution, symbols, risk_config, state_store, circuit_breaker, logger)
        self.df = data_df
        self.initial_equity = Decimal(str(initial_equity))
        self.is_oos = is_oos
        self.train_result = train_result
        self.risk_manager = risk_manager
        self.mode = "backtest"
        self.start_time = time.monotonic()
        self.trades = []
        self.realized_pnl = Decimal("0")
        self.warnings = []

    def run(self):
        df = self.df
        symbol = self.symbols[0]  # v1: single symbol
        min_bars = self.strategy.min_bars

        self.logger.info(
            f"Backtest started: {symbol}, {len(df)} bars, "
            f"initial_equity={self.initial_equity}"
        )

        engine = self.execution
        peak_equity = self.initial_equity
        daily_pnl = Decimal("0")
        current_day = ""

        for i in range(min_bars, len(df)):
            current_close = Decimal(str(df.iloc[i]["close"]))
            current_timestamp = int(df.iloc[i]["timestamp"])

            # Determine current day for daily PnL tracking
            bar_day = datetime.fromtimestamp(current_timestamp / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
            if bar_day != current_day:
                if current_day != "":
                    pass  # New day, daily_pnl carries over for circuit check
                current_day = bar_day

            # Update engine context
            engine.set_last_close(current_close)

            # === STEP 1: Exit checks (if position exists) ===
            position = engine.get_current_position(symbol)
            if position is not None:
                # Check SL/TP
                if self.risk_manager:
                    current_atr = self._calculate_atr(df, i, period=14)
                    sl_result, tp_result = self.risk_manager.check_exits(
                        {"entry_price": str(position.entry_price),
                         "high_since_entry": str(max(float(position.entry_price), float(current_close)))},
                        current_close,
                        current_atr,
                    )

                    if sl_result.triggered:
                        self.logger.warning(f"STOP LOSS triggered at {current_close}")
                        if i + 1 < len(df):
                            execution_price = Decimal(str(df.iloc[i + 1]["open"]))
                            engine.set_last_close(execution_price)
                        # Create sell signal for SL
                        sl_signal = type('Signal', (), {
                            'action': SignalAction.SELL,
                            'symbol': symbol,
                            'strength': 1.0,
                            'price': current_close,
                            'timestamp': current_timestamp,
                            'reason': f"Stop loss: {sl_result.stop_price}",
                        })()
                        order = engine.execute_signal(sl_signal, engine.get_equity(), position)
                        self.trades.append(order)
                        position = None
                        engine.update_equity_curve(current_close)
                        continue  # Skip to next bar

                    if tp_result.triggered:
                        self.logger.info(f"TAKE PROFIT triggered at {current_close}")
                        if i + 1 < len(df):
                            execution_price = Decimal(str(df.iloc[i + 1]["open"]))
                            engine.set_last_close(execution_price)
                        tp_signal = type('Signal', (), {
                            'action': SignalAction.SELL,
                            'symbol': symbol,
                            'strength': 1.0,
                            'price': current_close,
                            'timestamp': current_timestamp,
                            'reason': f"Take profit: {tp_result.tp_price}",
                        })()
                        order = engine.execute_signal(tp_signal, engine.get_equity(), position)
                        self.trades.append(order)
                        position = None
                        engine.update_equity_curve(current_close)
                        continue

            # === STEP 2: Signal evaluation (if no position) ===
            if position is None:
                window = df.iloc[:i + 1].copy()  # CRITICAL: only data up to bar i
                signal = self._process_bar(symbol, window)

                if signal.action == SignalAction.BUY:
                    # T+1 execution: use NEXT bar's open
                    if i + 1 < len(df):
                        execution_price = Decimal(str(df.iloc[i + 1]["open"]))
                        engine.set_last_close(execution_price)

                        # Risk check
                        if self.risk_manager:
                            if not self.risk_manager.check_signal(signal, engine.get_equity(), None):
                                self.logger.info(f"Signal rejected by risk manager")
                            else:
                                order = engine.execute_signal(signal, engine.get_equity(), None)
                                self.trades.append(order)
                                self.risk_manager.record_trade()
                    # else: last bar, discard signal

            # === STEP 3: Update equity curve ===
            engine.update_equity_curve(current_close)
            current_equity = engine.get_equity()

            if current_equity > peak_equity:
                peak_equity = current_equity

            # === STEP 4: Circuit breaker check ===
            if self.risk_manager:
                if not self.risk_manager.update_equity(current_equity, daily_pnl, peak_equity):
                    self.logger.critical("Circuit breaker tripped during backtest!")
                    self.warnings.append("Circuit breaker tripped")
                    break

        # === Post-backtest checks ===
        result = engine.get_result(self.initial_equity)
        total_trades = len([t for t in self.trades if t.side in ("buy", "sell")])

        if total_trades < 30:
            msg = f"交易次数不足 ({total_trades} < 30)，回测结果不可信"
            self.logger.warning(msg)
            self.warnings.append(msg)
        if total_trades < 10:
            self.logger.error(f"交易次数过少 ({total_trades} < 10)，拒绝输出指标")

        # Overfitting check
        if len(df) < min_bars * 2:
            msg = f"数据覆盖不足: {len(df)} bars < min_bars×2 ({min_bars * 2})"
            self.logger.warning(msg)
            self.warnings.append(msg)

        if total_trades > 0:
            winning = sum(1 for t in self.trades if t.side == "sell" and
                         t.filled_price and t.filled_price > Decimal("0"))
            # Approximate win rate from buy/sell pairs
            if len(self.trades) >= 2:
                # Count profitable round-trips
                buy_orders = [t for t in self.trades if t.side == "buy"]
                sell_orders = [t for t in self.trades if t.side == "sell"]
                profits = 0
                for j in range(min(len(buy_orders), len(sell_orders))):
                    buy_val = float(buy_orders[j].filled_quantity) * float(buy_orders[j].filled_price) if buy_orders[j].filled_price else 0
                    sell_val = float(sell_orders[j].filled_quantity) * float(sell_orders[j].filled_price) if sell_orders[j].filled_price else 0
                    if sell_val > buy_val:
                        profits += 1
                wr = profits / min(len(buy_orders), len(sell_orders)) if min(len(buy_orders), len(sell_orders)) > 0 else 0
                if wr < 0.2 or wr > 0.95:
                    msg = f"胜率极端 ({wr:.1%})，可能存在数据泄露或过拟合"
                    self.logger.warning(msg)
                    self.warnings.append(msg)

        self.logger.info(f"Backtest complete: {len(df)} bars, {total_trades} trades")
        return result

    def _calculate_atr(self, df, current_idx: int, period: int = 14):
        """Calculate ATR at current bar."""
        if current_idx < period:
            return None
        window = df.iloc[max(0, current_idx - period):current_idx + 1]
        high = window["high"].astype(float)
        low = window["low"].astype(float)
        close = window["close"].astype(float)

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.iloc[-period:].mean()
        return Decimal(str(atr))

    def shutdown(self):
        """Clean shutdown."""
        pass
