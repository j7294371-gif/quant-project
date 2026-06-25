.PHONY: install test backtest backtest-all paper live-dry lint docker-build docker-backtest

install:
	pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

test:
	pytest tests/ -v --ignore=tests/test_integration.py

test-integration:
	pytest tests/test_integration.py -v

test-all:
	pytest tests/ -v

# 回测（--start 可选——不传则从缓存最早数据开始）
backtest:
	python3 main.py backtest --strategy sma_cross --symbol BTC/USDT --timeframe 1h --start 2024-01-01

backtest-all:
	python3 main.py backtest --strategy sma_cross --symbol BTC/USDT --timeframe 1h --start 2024-01-01
	python3 main.py backtest --strategy macd --symbol BTC/USDT --timeframe 1h --start 2024-01-01
	python3 main.py backtest --strategy rsi --symbol BTC/USDT --timeframe 1h --start 2024-01-01
	python3 main.py backtest --strategy bollinger --symbol BTC/USDT --timeframe 1h --start 2024-01-01

paper:
	python3 main.py paper --strategy sma_cross

live-dry:
	python3 main.py live --strategy sma_cross --dry-run

lint:
	python3 -m ruff check src/

docker-build:
	docker-compose build

docker-backtest:
	docker-compose run --rm quant python3 main.py backtest --strategy sma_cross --symbol BTC/USDT --timeframe 1h --start 2024-01-01
