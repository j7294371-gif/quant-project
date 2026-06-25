import os
import sys
from pathlib import Path
from decimal import Decimal
from pydantic import BaseModel, Field, model_validator
import yaml
from dotenv import load_dotenv


class AppYamlConfig(BaseModel, frozen=True):
    log_level: str = "INFO"
    log_dir: str = "./logs"
    log_retention_days: int = 30
    data_cache_dir: str = "./data"
    state_dir: str = "./state"
    analytics: dict = Field(default_factory=lambda: {"risk_free_rate": 0.02})


class StrategyParams(BaseModel, frozen=True):
    pass  # params are validated per-strategy at runtime


class StrategyConfig(BaseModel, frozen=True):
    active: str
    sma_cross: dict = Field(default_factory=dict)
    macd: dict = Field(default_factory=dict)
    rsi: dict = Field(default_factory=dict)
    bollinger: dict = Field(default_factory=dict)
    fusion: dict = Field(default_factory=dict)

    @property
    def params(self) -> dict:
        """Return params for the active strategy."""
        strategy_params = getattr(self, self.active, {})
        return dict(strategy_params)


class StopLossConfig(BaseModel, frozen=True):
    type: str = "atr"
    fixed_pct: Decimal = Decimal("0.05")
    atr_multiplier: Decimal = Decimal("2.0")
    trailing_pct: Decimal = Decimal("0.03")


class TakeProfitConfig(BaseModel, frozen=True):
    type: str = "fixed_pct"
    fixed_pct: Decimal = Decimal("0.10")
    atr_multiplier: Decimal = Decimal("4.0")


class PortfolioSubConfig(BaseModel, frozen=True):
    lookback_days: int = 30
    correlation_high: Decimal = Decimal("0.8")
    correlation_extreme: Decimal = Decimal("0.95")


class RiskConfig(BaseModel, frozen=True):
    max_position_pct: Decimal = Decimal("0.2")
    max_drawdown_limit: Decimal = Decimal("0.3")
    daily_loss_limit: Decimal = Decimal("0.05")
    circuit_cooldown_hours: int = 24
    position_method: str = "fixed_pct"
    risk_per_trade_pct: Decimal = Decimal("0.01")
    stop_loss: StopLossConfig = Field(default_factory=StopLossConfig)
    take_profit: TakeProfitConfig = Field(default_factory=TakeProfitConfig)
    portfolio: PortfolioSubConfig = Field(default_factory=PortfolioSubConfig)

    @model_validator(mode="after")
    def validate_percentages(self):
        for field_name in ["max_position_pct", "max_drawdown_limit", "daily_loss_limit", "risk_per_trade_pct"]:
            val = getattr(self, field_name)
            if not (Decimal("0") < val <= Decimal("1")):
                raise ValueError(f"{field_name} 必须在 0.0 ~ 1.0 范围内，当前值: {val}")
        return self


class ExchangeConfig(BaseModel, frozen=True):
    exchange: str = "binance"
    testnet: bool = True
    symbols: list[str] = Field(default_factory=lambda: ["BTC/USDT", "ETH/USDT"])
    timeframe: str = "1h"

    @model_validator(mode="after")
    def validate_symbols(self):
        for s in self.symbols:
            if "/" not in s or not s.strip():
                raise ValueError(f"symbols 必须包含 '/' 分隔符且非空，无效值: '{s}'")
        return self


class AppConfig(BaseModel, frozen=True):
    app: AppYamlConfig
    strategy: StrategyConfig
    risk: RiskConfig
    exchange: ExchangeConfig
    api_key: str = ""
    api_secret: str = ""

    @classmethod
    def load(cls, config_dir: str = "./config") -> "AppConfig":
        # 1. Load .env
        load_dotenv()

        config_path = Path(config_dir)
        if not config_path.exists():
            print(f"错误: 配置目录不存在: {config_dir}", file=sys.stderr)
            sys.exit(1)

        def _read_yaml(filename: str) -> dict:
            filepath = config_path / filename
            if not filepath.exists():
                print(f"错误: 配置文件缺失: {filepath}", file=sys.stderr)
                sys.exit(1)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                print(f"错误: YAML 解析失败 ({filepath}): {e}", file=sys.stderr)
                sys.exit(1)

        try:
            # 2-5. Read YAML configs
            app_yaml = AppYamlConfig(**_read_yaml("app.yaml"))
            strategy_yaml = StrategyConfig(**_read_yaml("strategy.yaml"))
            risk_yaml = RiskConfig(**_read_yaml("risk.yaml"))
            exchange_yaml = ExchangeConfig(**_read_yaml("exchange.yaml"))

            # 6. Read env vars
            api_key = os.environ.get("EXCHANGE_API_KEY", "")
            api_secret = os.environ.get("EXCHANGE_API_SECRET", "")

            # 7. Aggregate
            config = cls(
                app=app_yaml,
                strategy=strategy_yaml,
                risk=risk_yaml,
                exchange=exchange_yaml,
                api_key=api_key,
                api_secret=api_secret,
            )
            return config

        except Exception as e:
            print(f"错误: 配置校验失败: {e}", file=sys.stderr)
            sys.exit(1)
