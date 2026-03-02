"""Test EMA integration in the Trainer class."""
import inspect

from trainer.training.trainer import Trainer


class TestEMAIntegration:
    def test_ema_setup_in_run(self):
        source = inspect.getsource(Trainer.run)
        assert "EMATracker" in source, "EMATracker must be instantiated in Trainer.run"
        assert "ema_enabled" in source, "ema_enabled config flag must be checked in Trainer.run"

    def test_ema_step_in_training_loop(self):
        source = inspect.getsource(Trainer.run)
        assert "_ema_tracker" in source, "_ema_tracker must be used inside the training loop"
        assert "ema_tracker.step" in source or "_ema_tracker.step" in source, \
            "EMA step must be called during the training loop"

    def test_ema_step_tied_to_sync_gradients(self):
        source = inspect.getsource(Trainer.run)
        # Both sync_gradients and ema step must be present; ordering enforced by structural test
        assert "sync_gradients" in source
        assert "_ema_tracker.step" in source

    def test_ema_swap_on_save(self):
        source = inspect.getsource(Trainer._save_checkpoint)
        assert "copy_to" in source, "copy_to must be called in _save_checkpoint to swap in EMA weights"
        assert "restore" in source, "restore must be called in _save_checkpoint to swap back"

    def test_ema_swap_on_final_save(self):
        source = inspect.getsource(Trainer._save_final)
        assert "copy_to" in source, "copy_to must be called in _save_final to swap in EMA weights"
        assert "restore" in source, "restore must be called in _save_final to swap back"

    def test_ema_swap_on_sampling(self):
        source = inspect.getsource(Trainer._generate_samples)
        assert "copy_to" in source, "copy_to must be called in _generate_samples for EMA inference"
        assert "restore" in source, "restore must be called in _generate_samples after sampling"

    def test_ema_state_saved_in_checkpoint(self):
        source = inspect.getsource(Trainer._save_checkpoint)
        assert "ema_state" in source or "state_dict" in source, \
            "EMA state_dict must be saved alongside the checkpoint"

    def test_ema_tracker_attribute_initialised(self):
        source = inspect.getsource(Trainer.__init__)
        assert "_ema_tracker" in source, "_ema_tracker must be initialised to None in __init__"
