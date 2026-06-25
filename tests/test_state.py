"""State persistence tests."""
import pytest
import json
import os
from decimal import Decimal


class TestStateStore:
    def test_save_load_roundtrip(self, temp_state_dir):
        from src.utils.state import StateStore
        store = StateStore(temp_state_dir)
        store.save("test", {"key": "value", "num": 42})
        loaded = store.load("test")
        assert loaded is not None
        assert loaded["key"] == "value"
        assert loaded["num"] == 42
        assert "updated_at" in loaded
        store.close()

    def test_load_nonexistent(self, temp_state_dir):
        from src.utils.state import StateStore
        store = StateStore(temp_state_dir)
        assert store.load("nonexistent") is None
        store.close()

    def test_delete_then_load(self, temp_state_dir):
        from src.utils.state import StateStore
        store = StateStore(temp_state_dir)
        store.save("test", {"x": 1})
        store.delete("test")
        assert store.load("test") is None
        store.close()

    def test_corrupted_json_returns_none(self, temp_state_dir):
        from src.utils.state import StateStore
        store = StateStore(temp_state_dir)
        # Write corrupted JSON
        path = os.path.join(temp_state_dir, "corrupt.json")
        with open(path, "w") as f:
            f.write("{invalid json")
        result = store.load("corrupt")
        assert result is None
        store.close()

    def test_clean_shutdown(self, temp_state_dir):
        from src.utils.state import StateStore
        store = StateStore(temp_state_dir)
        store.mark_shutdown("clean")
        assert store.is_clean_shutdown()
        store.close()

    def test_unclean_shutdown(self, temp_state_dir):
        from src.utils.state import StateStore
        store = StateStore(temp_state_dir)
        store.mark_shutdown("running")
        assert not store.is_clean_shutdown()
        store.close()

    def test_no_shutdown_file(self, temp_state_dir):
        from src.utils.state import StateStore
        store = StateStore(temp_state_dir)
        assert not store.is_clean_shutdown()
        store.close()

    def test_append_equity(self, temp_state_dir):
        from src.utils.state import StateStore
        store = StateStore(temp_state_dir)
        store.append_equity(1704067200000, Decimal("10000"))
        store.append_equity(1704070800000, Decimal("10100"))
        history = store.get_equity_history(1704067200000)
        assert len(history) == 2
        assert history[0][1] == Decimal("10000")
        store.close()
