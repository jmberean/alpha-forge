
import pytest
import os
import tempfile
from alphaforge.api.storage import Storage

class TestPersistence:
    @pytest.fixture
    def db_path(self):
        # Create temp file
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        yield path
        # Cleanup
        if os.path.exists(path):
            os.unlink(path)

    def test_save_and_load_validation(self, db_path):
        """Test saving and loading validation results."""
        storage = Storage(db_path=db_path)
        
        # Save
        data = {
            "strategy_name": "Test Strategy",
            "status": "completed",
            "passed": True,
            "metrics": {"sharpe": 1.5},
            "equity_curve": [{"date": "2023-01-01", "equity": 100}],
            "logs": ["Log 1"],
            "timestamp": "2023-01-01T00:00:00"
        }
        storage.save_validation("val_123", data)
        
        # Load
        loaded = storage.get_validation("val_123")
        assert loaded is not None
        assert loaded["validation_id"] == "val_123"
        assert loaded["strategy_name"] == "Test Strategy"
        assert loaded["metrics"]["sharpe"] == 1.5
        
        # List
        all_vals = storage.list_validations()
        assert len(all_vals) == 1
        assert all_vals[0]["validation_id"] == "val_123"

    def test_persistence_across_restarts(self, db_path):
        """Test data survives 'restart' (new Storage instance)."""
        # Run 1
        s1 = Storage(db_path=db_path)
        s1.save_validation("val_456", {"strategy_name": "Restart Test", "passed": False})
        
        # Run 2 (Simulate restart)
        s2 = Storage(db_path=db_path)
        loaded = s2.get_validation("val_456")
        
        assert loaded is not None
        assert loaded["strategy_name"] == "Restart Test"
