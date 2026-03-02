"""Tests for regularization / prior preservation (Task 22)."""
from __future__ import annotations

from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# RegDataIterator tests
# ---------------------------------------------------------------------------

class _MockDataLoader:
    """Minimal DataLoader stand-in backed by a list of batches."""

    def __init__(self, batches: list):
        self._batches = batches

    def __iter__(self):
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)


def test_reg_data_iterator_import():
    """RegDataIterator can be imported without error."""
    from trainer.data.loader import RegDataIterator  # noqa: F401


def test_reg_data_iterator_returns_batch_dict():
    """next_batch() returns the batch dict provided by the dataloader."""
    from trainer.data.loader import RegDataIterator

    batch = {"pixel_values": [1, 2, 3], "caption": "a dog"}
    loader = _MockDataLoader([batch])
    reg_iter = RegDataIterator(loader)
    result = reg_iter.next_batch()
    assert result == batch


def test_reg_data_iterator_cycles():
    """Iterator cycles back to the beginning after exhausting the dataloader."""
    from trainer.data.loader import RegDataIterator

    batches = [{"id": 0}, {"id": 1}]
    loader = _MockDataLoader(batches)
    reg_iter = RegDataIterator(loader)

    # First pass
    assert reg_iter.next_batch() == {"id": 0}
    assert reg_iter.next_batch() == {"id": 1}
    # Should cycle — next call should return first batch again
    assert reg_iter.next_batch() == {"id": 0}
    assert reg_iter.next_batch() == {"id": 1}


def test_reg_data_iterator_single_batch():
    """Single-batch dataloader keeps returning the same batch on cycle."""
    from trainer.data.loader import RegDataIterator

    batch = {"id": 0}
    loader = _MockDataLoader([batch])
    reg_iter = RegDataIterator(loader)

    for _ in range(5):
        assert reg_iter.next_batch() == batch


def test_reg_data_iterator_len():
    """__len__ returns the number of batches in the underlying dataloader."""
    from trainer.data.loader import RegDataIterator

    batches = [{"id": i} for i in range(7)]
    loader = _MockDataLoader(batches)
    reg_iter = RegDataIterator(loader)
    assert len(reg_iter) == 7


def test_reg_data_iterator_lazy_init():
    """The internal iterator is not created until next_batch() is first called."""
    from trainer.data.loader import RegDataIterator

    loader = _MockDataLoader([{"id": 0}])
    reg_iter = RegDataIterator(loader)
    assert reg_iter._iter is None  # not yet initialised
    reg_iter.next_batch()
    assert reg_iter._iter is not None


# ---------------------------------------------------------------------------
# DataConfig field tests
# ---------------------------------------------------------------------------

def test_data_config_has_reg_data_path():
    """DataConfig.reg_data_path exists and defaults to None."""
    from trainer.config.schema import DataConfig

    cfg = DataConfig()
    assert hasattr(cfg, "reg_data_path")
    assert cfg.reg_data_path is None


def test_data_config_has_prior_loss_weight():
    """DataConfig.prior_loss_weight exists and defaults to 1.0."""
    from trainer.config.schema import DataConfig

    cfg = DataConfig()
    assert hasattr(cfg, "prior_loss_weight")
    assert cfg.prior_loss_weight == 1.0


def test_data_config_reg_fields_round_trip():
    """reg_data_path and prior_loss_weight survive a model_dump / model_validate round-trip."""
    from trainer.config.schema import DataConfig

    cfg = DataConfig(reg_data_path="/data/reg", prior_loss_weight=0.5)
    dumped = cfg.model_dump()
    restored = DataConfig.model_validate(dumped)

    assert restored.reg_data_path == "/data/reg"
    assert restored.prior_loss_weight == 0.5


def test_data_config_custom_prior_loss_weight():
    """prior_loss_weight accepts custom float values."""
    from trainer.config.schema import DataConfig

    cfg = DataConfig(prior_loss_weight=0.75)
    assert cfg.prior_loss_weight == 0.75


def test_data_config_reg_data_path_accepts_string():
    """reg_data_path accepts a string path."""
    from trainer.config.schema import DataConfig

    cfg = DataConfig(reg_data_path="/some/path/reg.toml")
    assert cfg.reg_data_path == "/some/path/reg.toml"


# ---------------------------------------------------------------------------
# Trainer integration smoke test (no GPU/accelerate)
# ---------------------------------------------------------------------------

def test_trainer_imports_reg_loader_conditionally():
    """Trainer module can be imported and reg_loader is available for conditional use."""
    import trainer.training.trainer  # noqa: F401
    from trainer.data.loader import RegDataIterator  # noqa: F401


def test_reg_data_iterator_used_in_training_loop():
    """Simulate training loop behaviour: reg_iter.next_batch() is called each step."""
    from trainer.data.loader import RegDataIterator

    call_log: list[int] = []

    class TrackingLoader:
        def __init__(self, batches):
            self._batches = batches
            self._calls = 0

        def __iter__(self):
            TrackingLoader._instance = self
            return iter(self._batches)

        def __len__(self):
            return len(self._batches)

    batches = [{"step": i} for i in range(3)]
    loader = TrackingLoader(batches)
    reg_iter = RegDataIterator(loader)

    results = [reg_iter.next_batch() for _ in range(6)]

    # Should have cycled: steps 0,1,2 then 0,1,2
    assert results[0] == {"step": 0}
    assert results[3] == {"step": 0}  # cycle restarted
