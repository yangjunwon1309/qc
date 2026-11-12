"""One-step discounted graph-critic update checks for FQL."""

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
QC_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, QC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agents.acfql import ACFQLAgent
from graphcritic.io import load_targets, save_targets
from graphcritic.types import TargetSet


def _array_leaves(tree):
    return jax.tree_util.tree_leaves(tree)


def test_h1_discounted_target_loads_and_fql_graph_update_regresses_without_target_update(tmp_path):
    dataset_id = "ogbench:puzzle-3x3-play-singletask-task3-v0:test"
    target_path = tmp_path / "targets.npz"
    targets = TargetSet(
        y=np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        valid=np.ones(4, dtype=np.bool_),
        h=1,
        unit="steps",
        graph_hash="c" * 64,
        dataset_id=dataset_id,
        source=np.ones(4, dtype=np.uint8),
        kind="discounted",
        loss_mask=np.ones(4, dtype=np.bool_),
        discount_gamma=0.99,
    )
    save_targets(targets, target_path)
    loaded = load_targets(
        target_path, expected_dataset_id=dataset_id, require_loss_mask=True,
    )
    assert loaded.h == 1
    assert loaded.kind == "discounted"

    config = {
        "agent_name": "acfql", "lr": 3e-4, "batch_size": 4,
        "actor_hidden_dims": (16, 16), "value_hidden_dims": (16, 16),
        "layer_norm": True, "actor_layer_norm": False,
        "discount": 0.99, "tau": 0.005, "q_agg": "mean", "alpha": 100.0,
        "num_qs": 2, "flow_steps": 2, "normalize_q_loss": False,
        "encoder": None, "horizon_length": 1, "action_chunking": False,
        "actor_type": "distill-ddpg", "actor_num_samples": 32,
        "critic_mode": "graph", "graph_target_kind": "discounted",
        "graph_target_unit": "steps", "graph_target_gamma": 0.99,
        "use_fourier_features": False, "fourier_feature_dim": 8,
        "weight_decay": 0.0,
    }
    observations = jnp.arange(4 * 6, dtype=jnp.float32).reshape(4, 6) / 10.0
    example_actions = jnp.zeros((4, 2), dtype=jnp.float32)
    agent = ACFQLAgent.create(0, observations, example_actions, config)
    critic_before = agent.network.params["modules_critic"]
    target_before = agent.network.params["modules_target_critic"]
    batch = {
        "observations": observations,
        "actions": jnp.zeros((4, 1, 2), dtype=jnp.float32),
        "valid": jnp.ones((4, 1), dtype=jnp.float32),
        "graph_target": jnp.asarray(loaded.y),
        "graph_target_valid": jnp.asarray(loaded.valid),
        "graph_target_loss_mask": jnp.asarray(loaded.loss_mask),
    }

    updated, info = agent.update(batch)

    assert float(info["critic/critic_loss"]) > 0.0
    assert np.isfinite(float(info["critic/discounted_mean"]))
    assert any(
        not np.array_equal(np.asarray(before), np.asarray(after))
        for before, after in zip(
            _array_leaves(critic_before),
            _array_leaves(updated.network.params["modules_critic"]),
            strict=True,
        )
    )
    for before, after in zip(
        _array_leaves(target_before),
        _array_leaves(updated.network.params["modules_target_critic"]),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(before), np.asarray(after))
