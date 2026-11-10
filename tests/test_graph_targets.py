"""Regret critic plumbing tests; run with `pixi run -e bench pytest ...`."""

from pathlib import Path
import sys

import pytest

pytest.importorskip("jax", reason="QC graph-target plumbing requires the bench environment")

import jax.numpy as jnp
import numpy as np

QC_ROOT = Path(__file__).resolve().parents[1]
if str(QC_ROOT) not in sys.path:
    sys.path.insert(0, str(QC_ROOT))

from agents.acfql import _masked_spearman
from main import _load_graph_target_provenance, _validate_oracle_run_group
from utils.datasets import Dataset


def test_masked_spearman_ignores_excluded_rows():
    value = _masked_spearman(
        jnp.asarray([1.0, 2.0, 3.0, 100.0]),
        jnp.asarray([3.0, 2.0, 1.0, -100.0]),
        jnp.asarray([True, True, True, False]),
    )
    np.testing.assert_allclose(np.asarray(value), -1.0, atol=1e-6)


def test_sequence_sampler_returns_graph_annotations_at_start_row(monkeypatch):
    n_rows = 8
    row_values = np.arange(n_rows, dtype=np.float32)
    dataset = Dataset.create(
        freeze=False,
        observations=row_values[:, None],
        next_observations=(row_values + 1)[:, None],
        actions=np.zeros((n_rows, 1), dtype=np.float32),
        rewards=np.zeros(n_rows, dtype=np.float32),
        masks=np.ones(n_rows, dtype=np.float32),
        terminals=np.asarray([0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float32),
        graph_target=10.0 + row_values,
        graph_target_valid=np.ones(n_rows, dtype=np.bool_),
        graph_target_loss_mask=np.asarray([1, 0, 1, 1, 0, 1, 1, 1], dtype=np.bool_),
    )
    monkeypatch.setattr(np.random, "randint", lambda high, size=None: np.asarray([1, 5]))

    batch = dataset.sample_sequence(batch_size=2, sequence_length=3, discount=0.99)

    np.testing.assert_array_equal(batch["graph_target"], [11.0, 15.0])
    np.testing.assert_array_equal(batch["graph_target_valid"], [True, True])
    np.testing.assert_array_equal(batch["graph_target_loss_mask"], [False, True])
    np.testing.assert_array_equal(batch["observations"][:, 0], [1.0, 5.0])


def test_legacy_target_metadata_defaults_to_learned(tmp_path):
    target_json = tmp_path / "targets.json"
    target_json.write_text('{"config": {}}')

    assert _load_graph_target_provenance(tmp_path / "targets.npz") == {"kind": "learned"}


def test_oracle_target_provenance_is_read_and_requires_oracle_run_group(tmp_path):
    target_json = tmp_path / "targets.json"
    target_json.write_text(
        '{"target_provenance": {"kind": "oracle_upper_bound", "oracle_np_max": 1}}'
    )
    provenance = _load_graph_target_provenance(tmp_path / "targets.npz")

    assert provenance == {"kind": "oracle_upper_bound", "oracle_np_max": 1}
    _validate_oracle_run_group(provenance, "v083-p13-d1-oracle")
    with pytest.raises(ValueError, match="require --run_group"):
        _validate_oracle_run_group(provenance, "v083-p13-d1")

