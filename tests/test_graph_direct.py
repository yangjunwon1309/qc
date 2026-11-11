"""Small update-path checks for the critic-free v0.9 QC mode."""

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

QC_ROOT = Path(__file__).resolve().parents[1]
if str(QC_ROOT) not in sys.path:
    sys.path.insert(0, str(QC_ROOT))

from agents.acfql import ACFQLAgent


def _array_leaves(tree):
    return jax.tree_util.tree_leaves(tree)


def test_graph_direct_update_keeps_critic_and_target_parameters_unchanged():
    config = {
        "agent_name": "acfql", "lr": 3e-4, "batch_size": 4,
        "actor_hidden_dims": (16, 16), "value_hidden_dims": (16, 16),
        "layer_norm": True, "actor_layer_norm": False,
        "discount": 0.99, "tau": 0.005, "q_agg": "mean", "alpha": 100.0,
        "num_qs": 2, "flow_steps": 2, "normalize_q_loss": False,
        "encoder": None, "horizon_length": 5, "action_chunking": True,
        "actor_type": "best-of-n", "actor_num_samples": 4,
        "critic_mode": "graph_direct", "use_fourier_features": False,
        "fourier_feature_dim": 8, "weight_decay": 0.1,
    }
    observations = jnp.zeros((4, 12), dtype=jnp.float32)
    actions = jnp.zeros((4, 2), dtype=jnp.float32)
    agent = ACFQLAgent.create(0, observations, actions, config)
    before_critic = agent.network.params["modules_critic"]
    before_target = agent.network.params["modules_target_critic"]
    batch = {
        "observations": observations,
        "actions": jnp.zeros((4, 5, 2), dtype=jnp.float32),
        "valid": jnp.ones((4, 5), dtype=jnp.float32),
    }
    updated, info = agent.update(batch)
    for before, after in zip(_array_leaves(before_critic), _array_leaves(updated.network.params["modules_critic"]), strict=True):
        np.testing.assert_array_equal(np.asarray(before), np.asarray(after))
    for before, after in zip(_array_leaves(before_target), _array_leaves(updated.network.params["modules_target_critic"]), strict=True):
        np.testing.assert_array_equal(np.asarray(before), np.asarray(after))
    assert float(info["critic/critic_loss"]) == 0.0

