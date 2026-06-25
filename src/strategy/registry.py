from src.strategy.base import BaseStrategy

STRATEGY_REGISTRY: dict[str, type[BaseStrategy]] = {}


def register_strategy(name: str):
    def decorator(cls: type[BaseStrategy]):
        if name in STRATEGY_REGISTRY:
            raise ValueError(
                f"策略名 '{name}' 已被注册（类: {STRATEGY_REGISTRY[name].__name__}），拒绝重复注册"
            )
        STRATEGY_REGISTRY[name] = cls
        return cls
    return decorator


def get_strategy(name: str, params: dict) -> BaseStrategy:
    if name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"未注册的策略: '{name}'。已注册: {list(STRATEGY_REGISTRY.keys())}"
        )
    return STRATEGY_REGISTRY[name](params)
