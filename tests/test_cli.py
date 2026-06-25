"""CLI argument parsing tests."""
import pytest
import sys


class TestCLIArgs:
    def test_parse_backtest_args(self):
        """Basic backtest args parse correctly."""
        sys.argv = ["main.py", "backtest", "--strategy", "sma_cross", "--symbol", "BTC/USDT"]
        from main import parse_args
        args = parse_args()
        assert args.command == "backtest"
        assert args.strategy == "sma_cross"
        assert args.symbol == "BTC/USDT"

    def test_oos_exclusive_with_start_end(self):
        """OOS and --start/--end are mutually exclusive."""
        sys.argv = [
            "main.py", "backtest",
            "--strategy", "sma_cross", "--symbol", "BTC/USDT",
            "--train-start", "2024-01-01", "--train-end", "2024-06-30",
            "--test-start", "2024-07-01", "--test-end", "2024-12-31",
            "--start", "2024-01-01",
        ]
        from main import parse_args, _validate_args
        args = parse_args()
        with pytest.raises(SystemExit):
            _validate_args(args)

    def test_partial_oos_params_error(self):
        """Partial OOS parameters cause error."""
        sys.argv = [
            "main.py", "backtest",
            "--strategy", "sma_cross", "--symbol", "BTC/USDT",
            "--train-start", "2024-01-01",
        ]
        from main import parse_args, _validate_args
        args = parse_args()
        with pytest.raises(SystemExit):
            _validate_args(args)

    def test_test_before_train_error(self):
        """--test-start < --train-end causes error."""
        sys.argv = [
            "main.py", "backtest",
            "--strategy", "sma_cross", "--symbol", "BTC/USDT",
            "--train-start", "2024-06-01", "--train-end", "2024-12-31",
            "--test-start", "2024-07-01", "--test-end", "2024-12-31",
        ]
        from main import parse_args, _validate_args
        args = parse_args()
        with pytest.raises(SystemExit):
            _validate_args(args)

    def test_list_strategies(self):
        """list-strategies subcommand works."""
        sys.argv = ["main.py", "list-strategies"]
        from main import parse_args
        args = parse_args()
        assert args.command == "list-strategies"

    def test_paper_mode(self):
        sys.argv = ["main.py", "paper", "--strategy", "sma_cross", "--symbols", "BTC/USDT", "ETH/USDT"]
        from main import parse_args
        args = parse_args()
        assert args.command == "paper"
        assert len(args.symbols) == 2

    def test_live_dry_run(self):
        sys.argv = ["main.py", "live", "--strategy", "sma_cross", "--dry-run"]
        from main import parse_args
        args = parse_args()
        assert args.command == "live"
        assert args.dry_run
