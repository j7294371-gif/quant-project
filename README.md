# Quant Project — 量化交易系统

多策略回测+模拟盘+实盘，Docker 一键部署。

## 快速开始（3 步）

1. 复制配置：
   ```bash
   cp .env.example .env
   # 编辑 .env，填入真实 API Key (EXCHANGE_API_KEY / EXCHANGE_API_SECRET)
   # 仅回测模式可留空 API Key
   ```

2. 安装依赖：
   ```bash
   pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
   # 或：pip install -e ".[dev]"
   ```

3. 运行回测：
   ```bash
   python3 main.py backtest --strategy sma_cross --symbol BTC/USDT --timeframe 1h --start 2024-01-01
   ```

## 运行模式

| 模式 | 命令 | 说明 |
|------|------|------|
| 回测 | `python3 main.py backtest --strategy sma_cross --symbol BTC/USDT --start 2024-01-01` | 历史数据回测 |
| 样本外测试 | `python3 main.py backtest --train-start 2024-01-01 --train-end 2024-06-30 --test-start 2024-07-01 --test-end 2024-12-31 --strategy sma_cross --symbol BTC/USDT` | 训练/测试分割 |
| 模拟盘 | `python3 main.py paper --strategy sma_cross` | WebSocket 实时模拟成交 |
| 实盘 | `python3 main.py live --strategy sma_cross --dry-run` | 先 dry-run 确认无 bug 再取消 `--dry-run` |
| 列出策略 | `python3 main.py list-strategies` | 查看所有可用的策略 |

## Docker 部署

```bash
make docker-build     # 构建镜像
make docker-backtest  # 运行回测
```

## 配置

所有配置文件在 `config/` 目录下，详见各 YAML 文件内注释。
API Key 通过 `.env` 文件（或环境变量）注入，绝不写入 YAML 或代码中。

## 目录结构

```
quant_project/
├── config/          # YAML 配置文件
├── data/            # CSV 历史数据缓存
├── logs/            # 日志 + 交易日志
├── state/           # SQLite 检查点持久化
├── src/             # 源代码（策略/风控/执行/分析/数据/组合管理）
├── tests/           # pytest 测试套件
├── main.py          # CLI 统一入口
├── Dockerfile       # 容器镜像
└── docker-compose.yml
```

## 免责声明

本软件仅用于教育和研究目的。量化交易有风险，作者不对任何交易损失负责。
使用前请确认你所在的地区允许加密货币交易。
