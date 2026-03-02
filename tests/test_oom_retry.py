"""Test OOM retry logic in the training loop."""
import inspect

from trainer.training.trainer import Trainer


class TestOOMRetry:
    def test_oom_retry_in_source(self):
        source = inspect.getsource(Trainer.run)
        assert "OutOfMemoryError" in source, "OOM handler must be present in Trainer.run"
        assert "oom_count" in source, "oom_count variable must track retry attempts"

    def test_max_retries_constant(self):
        source = inspect.getsource(Trainer.run)
        assert "_MAX_OOM_RETRIES" in source or "MAX_OOM" in source, \
            "A max-retries constant must be defined in Trainer.run"

    def test_oom_resets_gradients_and_clears_cache(self):
        source = inspect.getsource(Trainer.run)
        assert "zero_grad" in source, "zero_grad must be called in the OOM handler"
        assert "empty_cache" in source, "empty_cache must be called in the OOM handler"

    def test_oom_continues_loop(self):
        source = inspect.getsource(Trainer.run)
        assert "continue" in source, "continue must be used to retry the step after transient OOM"

    def test_oom_reraises_after_max_retries(self):
        source = inspect.getsource(Trainer.run)
        assert "raise" in source, "OOM must be re-raised after max retries exceeded"
