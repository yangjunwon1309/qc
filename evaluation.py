from collections import defaultdict

import jax
import numpy as np
from tqdm import trange
from functools import partial


def _live_puzzle_readout(env):
    """Read qpos and exact board bits from the live MuJoCo environment."""
    state = {}
    getter = getattr(env, 'get_state', None)
    if callable(getter):
        value = getter()
        if isinstance(value, dict):
            state.update(value)
    unwrapped = getattr(env, 'unwrapped', env)
    data = getattr(unwrapped, 'data', None)
    if 'qpos' not in state and data is not None:
        state['qpos'] = np.asarray(data.qpos).copy()
    if 'button_states' not in state and hasattr(unwrapped, '_cur_button_states'):
        state['button_states'] = np.asarray(unwrapped._cur_button_states).copy()
    if 'qpos' not in state or 'button_states' not in state:
        raise ValueError('direct-Q evaluation requires live qpos and button_states')
    return np.asarray(state['qpos']).copy(), np.asarray(state['button_states']).copy()


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, rng=key, **kwargs)

    return wrapped


def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)

def evaluate(
    agent,
    env,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0,
    eval_gaussian=None,
    action_shape=None,
    observation_shape=None,
    action_dim=None,
    action_selector=None,
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        eval_temperature: Action sampling temperature.
        eval_gaussian: Standard deviation of the Gaussian noise to add to the actions.

    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    trajs = []
    stats = defaultdict(list)
    selector_diagnostics = []

    renders = []
    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes

        observation, info = env.reset()
            
        observation_history = []
        observation_history.append(np.asarray(observation).copy())
        physical_history = []
        if action_selector is not None:
            physical_history.append(_live_puzzle_readout(env))
        action_history = []
        
        done = False
        step = 0
        render = []
        action_chunk_lens = defaultdict(lambda: 0)

        action_queue = []

        gripper_contact_lengths = []
        gripper_contact_length = 0
        while not done:
            
            if action_selector is None:
                action = actor_fn(observations=observation)

            if len(action_queue) == 0:
                have_new_action = True
                if action_selector is not None:
                    action, selection_info = action_selector(agent, observation, observation_history, physical_history)
                    if selection_info is not None:
                        selector_diagnostics.append(selection_info)
                action = np.array(action).reshape(-1, action_dim)
                action_chunk_len = action.shape[0]
                for a in action:
                    action_queue.append(a)
            else:
                have_new_action = False
            
            action = action_queue.pop(0)
            if eval_gaussian is not None:
                action = np.random.normal(action, eval_gaussian)

            next_observation, reward, terminated, truncated, info = env.step(np.clip(action, -1, 1))
            done = terminated or truncated
            step += 1

            if should_render and (step % video_frame_skip == 0 or done):
                frame = env.render().copy()
                render.append(frame)

            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)
            
            observation = next_observation
            observation_history.append(np.asarray(observation).copy())
            if action_selector is not None:
                physical_history.append(_live_puzzle_readout(env))
            # print(info)
            if "proprio" in info and "gripper_contact" in info["proprio"]:
                # print(info["gripper_contact"])
                gripper_contact = info["proprio"]["gripper_contact"]
            elif "gripper_contact" in info:
                gripper_contact = info["gripper_contact"]
            else:
                gripper_contact = None

            if gripper_contact is not None:
                if info["gripper_contact"] > 0.1:
                    gripper_contact_length += 1
                else:
                    if gripper_contact_length > 0:
                        gripper_contact_lengths.append(gripper_contact_length)
                    gripper_contact_length = 0

        if gripper_contact_length > 0:
            gripper_contact_lengths.append(gripper_contact_length)
        
        num_gripper_contacts = len(gripper_contact_lengths)

        if num_gripper_contacts > 0:
            avg_gripper_contact_length = np.mean(np.array(gripper_contact_lengths))
        else:
            avg_gripper_contact_length = 0
            
        add_to(stats, {"avg_gripper_contact_length": avg_gripper_contact_length, "num_gripper_contacts": num_gripper_contacts})

        if i < num_eval_episodes:
            add_to(stats, flatten(info))
            trajs.append(traj)
        else:
            renders.append(np.array(render))

    for k, v in stats.items():
        stats[k] = np.mean(v)

    if selector_diagnostics:
        stats['direct_q/fallback_fraction'] = float(np.mean([d.get('fallback', False) for d in selector_diagnostics]))
        stats['direct_q/mean_neighborhood_size'] = float(np.mean([d.get('neighborhood_size', 0) for d in selector_diagnostics]))
        selected_q = [d['selected_q'] for d in selector_diagnostics if d.get('selected_q') is not None and np.isfinite(d['selected_q'])]
        q_std = [d['candidate_q_std'] for d in selector_diagnostics if d.get('candidate_q_std') is not None and np.isfinite(d['candidate_q_std'])]
        stats['direct_q/selected_q_mean'] = float(np.mean(selected_q)) if selected_q else np.nan
        stats['direct_q/candidate_q_std_mean'] = float(np.mean(q_std)) if q_std else np.nan
        stats['direct_q/decision_seconds_mean'] = float(np.mean([d.get('decision_seconds', 0.0) for d in selector_diagnostics]))
        reasons = {}
        for diagnostic in selector_diagnostics:
            if diagnostic.get('fallback'):
                key = str(diagnostic.get('fallback_reason') or 'unknown')
                reasons[key] = reasons.get(key, 0) + 1
        stats['direct_q/fallback_count'] = int(sum(reasons.values()))
        for key in ('no_board_nodes', 'no_finite_cost_nodes', 'no_route_within_K', 'unknown'):
            stats[f'direct_q/fallback_{key}'] = int(reasons.get(key, 0))
    if 'episode.final_reward' in stats:
        stats['episode.final_wrong_buttons'] = -float(stats['episode.final_reward'])

    return stats, trajs, renders

