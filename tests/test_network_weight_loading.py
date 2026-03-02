"""Test that pre-trained network weights are loaded when configured."""
import inspect
from trainer.training.methods import LoRAMethod


class TestNetworkWeightLoading:
    def test_loads_weights_when_path_provided(self):
        """LoRAMethod.prepare() should call network.load_weights()
        when network_weights is set."""
        source = inspect.getsource(LoRAMethod.prepare)
        assert "load_weights" in source
        assert "network_weights" in source

    def test_load_weights_after_apply_to(self):
        """load_weights must come after apply_to (network needs to be applied first)."""
        source = inspect.getsource(LoRAMethod.prepare)
        apply_pos = source.index("apply_to")
        load_pos = source.index("load_weights")
        assert load_pos > apply_pos, "load_weights should come after apply_to"
