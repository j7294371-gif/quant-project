import time
import queue
from decimal import Decimal
from datetime import datetime, timezone
from src.orchestrator.base import BaseOrchestrator
from src.strategy.base import SignalAction


class PaperOrchestrator(BaseOrchestrator):
    def __init__(
        self,
        strategy,
        execution,
        symbols: list[str],
        risk_config,
        state_store,
        circuit_breaker,
        logger,
        stream,
        risk_manager=None,
    ):
        super().__init__(strategy, execution, symbols, risk_config, state_store, circuit_breaker, logger)
        self.stream = stream
        self.risk_manager = risk_manager
        self.mode = "paper"
        self.start_time = time.monotonic()
        self.trades = []
        self.realized_pnl = Decimal("0")
        self.warnings = []
        self._last_bar_timestamps: dict[str, int] = {}
        self._dataframes: dict[str, list] = {s: [] for s in symbols}
        self._stop_event = False

    def run(self):
        self.logger.info(f"Paper trading started: {self.symbols}")
        self.stream.start()

        consecutive_empty = 0
        peak_equity = Decimal("10000")
        last_health_check = time.monotonic()

        try:
            while not self._stop_event:
                try:
                    symbol, ohlcv = self.stream.get(timeout=60.0)

                    # Check for error from stream thread
                    if symbol == "__error__":
                        self.logger.error(f"Stream error: {ohlcv}")
                        self.circuit.trip("data_stream_error")
                        break

                    # CRITICAL: Reset counter on successful data
                    consecutive_empty = 0

                    self._last_bar_timestamps[symbol] = int(ohlcv.iloc[-1]["timestamp"])

                    # Accumulate data and process
                    self._dataframes[symbol].append(ohlcv)
                    # Keep last 500 bars
                    if len(self._dataframes[symbol]) > 500:
                        self._dataframes[symbol] = self._dataframes[symbol][-500:]

                    import pandas as pd
                    df = pd.concat(self._dataframes[symbol], ignore_index=True)

                    self._process_paper_bar(symbol, df, peak_equity)

                except queue.Empty:
                    consecutive_empty += 1
                    self.logger.warning(
                        f"WebSocket 60s 无数据 (连续第 {consecutive_empty} 次) | "
                        f"最后 K 线: {self._last_bar_timestamps}"
                    )
                    if consecutive_empty >= 5:
                        self.logger.error("数据流中断 >= 5 分钟，触发紧急熔断")
                        self.circuit.trip("data_stream_timeout")
                        self.execution.close_all_positions("data_stream_timeout")
                        break

                # Hourly health check
                now = time.monotonic()
                if now - last_health_check >= 3600:
                    from src.analytics.report import print_health_check
                    print_health_check(
                        start_time=self.start_time,
                        mode=self.mode,
                        symbols=self.symbols,
                        total_trades=len(self.trades),
                        daily_pnl=self.realized_pnl,
                        circuit_breaker=self.circuit,
                        ws_connected=consecutive_empty == 0,
                        last_bar_time=max(self._last_bar_timestamps.values()) if self._last_bar_timestamps else 0,
                    )
                    last_health_check = now

        except KeyboardInterrupt:
            self.logger.info("Paper mode interrupted by user")
        finally:
            self.stream.stop()

        return None

    def _process_paper_bar(self, symbol: str, df, peak_equity: Decimal):
        """Process one bar: strategy -> risk -> execute."""
        import pandas as pd

        signal = self._process_bar(symbol, df)
        if signal.action == SignalAction.HOLD:
            return

        position = self.execution.get_current_position(symbol)
        equity = self.execution.get_equity()

        # SELL signals bypass risk checks
        if signal.action == SignalAction.SELL:
            order = self.execution.execute_signal(signal, equity, position)
            self.trades.append(order)
            return

        # BUY signals: risk checks
        if self.risk_manager:
            if not self.risk_manager.check_signal(signal, equity, position):
                return

        order = self.execution.execute_signal(signal, equity, position)
        self.trades.append(order)

        if self.risk_manager:
            self.risk_manager.record_trade()

        # Update peak equity
        if order.side == "sell" and order.filled_price:
            sell_val = float(order.filled_quantity) * float(order.filled_price)
            buy_val = float(position.quantity) * float(position.entry_price) if position else 0
            pnl = Decimal(str(sell_val - buy_val - float(order.fee)))
            self.realized_pnl += pnl

    def shutdown(self):
        self._stop_event = True
        self.stream.stop()
