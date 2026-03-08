"""Tests for ConfigBinder - flat/nested dict roundtrips and callbacks."""
from ui.binding import ConfigBinder


SAMPLE_NESTED = {
    "model": {
        "architecture": "wan",
        "base_model_path": "/models/wan",
        "dtype": "bf16",
    },
    "training": {
        "method": "lora",
        "batch_size": 2,
        "epochs": 10,
    },
    "data": {
        "datasets": [{"path": "/data/train", "repeats": 5}],
        "resolution": 512,
    },
}


class TestConfigBinder:
    def test_flatten_unflatten_roundtrip(self):
        binder = ConfigBinder()
        binder.load_from_dict(SAMPLE_NESTED)
        result = binder.to_config_dict()
        assert result == SAMPLE_NESTED

    def test_load_from_dict(self):
        binder = ConfigBinder()
        binder.load_from_dict(SAMPLE_NESTED)
        assert binder.get("model.architecture") == "wan"
        assert binder.get("training.batch_size") == 2
        assert binder.get("data.resolution") == 512

    def test_to_config_dict(self):
        binder = ConfigBinder()
        binder.load_from_dict({"a": {"b": {"c": 42}}})
        result = binder.to_config_dict()
        assert result == {"a": {"b": {"c": 42}}}

    def test_set_fires_callback(self):
        binder = ConfigBinder()
        binder.load_from_dict(SAMPLE_NESTED)

        changes = []
        binder.on_change(lambda path, value: changes.append((path, value)))

        binder.set("training.batch_size", 4)
        assert changes == [("training.batch_size", 4)]
        assert binder.get("training.batch_size") == 4

    def test_update_many_skips_callbacks(self):
        binder = ConfigBinder()
        binder.load_from_dict(SAMPLE_NESTED)

        changes = []
        binder.on_change(lambda path, value: changes.append((path, value)))

        binder.update_many({
            "training.batch_size": 8,
            "model.dtype": "fp16",
        })
        assert changes == []
        assert binder.get("training.batch_size") == 8
        assert binder.get("model.dtype") == "fp16"

    def test_list_values_not_expanded(self):
        binder = ConfigBinder()
        binder.load_from_dict(SAMPLE_NESTED)
        # Lists must remain as leaf values, not flattened into dotted indices
        datasets = binder.get("data.datasets")
        assert isinstance(datasets, list)
        assert len(datasets) == 1
        assert datasets[0]["path"] == "/data/train"

    def test_get_default(self):
        binder = ConfigBinder()
        assert binder.get("nonexistent", "default") == "default"
        assert binder.get("nonexistent") is None

    def test_keys(self):
        binder = ConfigBinder()
        binder.load_from_dict({"a": {"b": 1}, "c": 2})
        keys = binder.keys()
        assert sorted(keys) == ["a.b", "c"]
