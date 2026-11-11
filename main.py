import glob, tqdm, wandb, os, json, random, time, jax
from pathlib import Path
from absl import app, flags
from ml_collections import config_flags
from log_utils import setup_wandb, get_exp_name, get_flag_dict, CsvLogger

from envs.env_utils import make_env_and_datasets
from envs.ogbench_utils import make_ogbench_env_and_datasets
from envs.robomimic_utils import is_robomimic_env

from utils.flax_utils import save_agent
from utils.datasets import Dataset, ReplayBuffer

from evaluation import evaluate
from agents import agents
import numpy as np

if 'CUDA_VISIBLE_DEVICES' in os.environ:
    os.environ['EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']
    os.environ['MUJOCO_EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-triple-play-singletask-task2-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('graph_targets', None, 'GraphCritic targets.npz/JSON pair for offline graph-critic training.')
flags.DEFINE_string('graph_dataset_id', None, 'Expected dataset ID recorded in the graph target metadata.')
flags.DEFINE_string('graph_target_provenance', None, 'Resolved provenance of graph targets; populated from target JSON.')
flags.DEFINE_string('direct_q_config', None, 'direct graph-Q scorer JSON configuration.')
flags.DEFINE_bool('eval_bc_also', False, 'Evaluate the same actor checkpoint with one unscored BC proposal.')

flags.DEFINE_integer('offline_steps', 1000000, 'Number of online steps.')
flags.DEFINE_integer('online_steps', 1000000, 'Number of online steps.')
flags.DEFINE_integer('buffer_size', 2000000, 'Replay buffer size.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', -1, 'Save interval.')
flags.DEFINE_integer('start_training', 5000, 'when does training start')

flags.DEFINE_integer('utd_ratio', 1, "update to data ratio")

flags.DEFINE_float('discount', 0.99, 'discount factor')

flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

config_flags.DEFINE_config_file('agent', 'agents/acfql.py', lock_config=False)

flags.DEFINE_float('dataset_proportion', 1.0, "Proportion of the dataset to use")
flags.DEFINE_integer('dataset_replace_interval', 1000, 'Dataset replace interval, used for large datasets because of memory constraints')
flags.DEFINE_string('ogbench_dataset_dir', None, 'OGBench dataset directory')
flags.DEFINE_string('ogbench_dataset_path', None, 'Fixed OGBench training NPZ; requires adjacent -val.npz')

flags.DEFINE_integer('horizon_length', 5, 'action chunking length.')
flags.DEFINE_bool('sparse', False, "make the task sparse reward")

flags.DEFINE_bool('save_all_online_states', False, "save all trajectories to npy")


def _qc_chunk_valid(terminals, horizon):
    """Return QC's start-row validity for its `valid[..., -1]` chunk mask."""
    terminals = np.asarray(terminals, dtype=np.bool_)
    n_rows = terminals.size
    valid = np.zeros(n_rows, dtype=np.bool_)
    n_starts = max(0, n_rows - horizon + 1)
    if n_starts == 0:
        return valid
    prefix = np.r_[0, np.cumsum(terminals, dtype=np.int64)]
    starts = np.arange(n_starts)
    # A terminal on the last row of the chunk is valid; only an earlier
    # terminal invalidates the endpoint used by QC's critic loss.
    valid[:n_starts] = (prefix[starts + horizon - 1] - prefix[starts]) == 0
    return valid


def _load_graph_target_provenance(target_path):
    """Read target provenance; legacy target files are treated as learned."""
    if target_path is None:
        return {'kind': 'learned'}
    metadata_path = Path(target_path).with_suffix('.json')
    try:
        metadata = json.loads(metadata_path.read_text())
    except FileNotFoundError:
        # load_targets will report a missing JSON artifact after environment setup.
        return {'kind': 'learned'}
    provenance = metadata.get('target_provenance')
    if provenance is None:
        return {'kind': 'learned'}
    if not isinstance(provenance, dict):
        raise ValueError('target_provenance must be a JSON object')
    kind = provenance.get('kind')
    if kind == 'learned':
        return {'kind': 'learned'}
    if kind == 'oracle_upper_bound':
        oracle_np_max = provenance.get('oracle_np_max')
        if isinstance(oracle_np_max, bool) or not isinstance(oracle_np_max, int) or oracle_np_max < 0:
            raise ValueError('oracle_upper_bound provenance requires a non-negative integer oracle_np_max')
        return {'kind': kind, 'oracle_np_max': oracle_np_max}
    raise ValueError(f'unsupported graph target provenance kind: {kind!r}')


def _validate_oracle_run_group(provenance, run_group):
    if provenance.get('kind') == 'oracle_upper_bound' and 'oracle' not in str(run_group).lower():
        raise ValueError(
            'oracle_upper_bound targets require --run_group to contain the word "oracle"'
        )


def _provenance_csv_value(provenance):
    if provenance.get('kind') == 'oracle_upper_bound':
        return f"oracle_upper_bound|oracle_np_max={provenance['oracle_np_max']}"
    return 'learned'


def _load_direct_q_provenance(config_path):
    if config_path is None:
        return None
    config = json.loads(Path(config_path).read_text())
    graph_path = Path(config['graph'])
    graph_meta = json.loads(graph_path.with_suffix('.json').read_text())
    meta = graph_meta.get('meta', {})
    provenance = meta.get('target_provenance', {})
    if graph_meta.get('content_hash') != config.get('graph_hash'):
        raise ValueError('direct-Q config graph hash does not match graph metadata')
    if config.get('spec_version') == 'v1.0':
        if config.get('anchor_mode') != 'dplus' or provenance.get('kind') != 'learned':
            raise ValueError('v1.0 graph_direct requires a learned L+ graph and dplus anchors')
        model_meta = meta.get('bridge_model_metadata', {})
        if model_meta.get('params_sha256') != config.get('state_checkpoint_sha256'):
            raise ValueError('L+ graph was not built from the configured dplus checkpoint')
        return {'kind': 'learned', 'graph_hash': config['graph_hash'], 'spec_version': 'v1.0'}
    if provenance.get('kind') != 'oracle_upper_bound' or provenance.get('oracle_np_max') != 1:
        raise ValueError('v0.9 graph_direct requires the O_np1 oracle_upper_bound graph')
    if meta.get('qc_eligible') != 'oracle_upper_bound':
        raise ValueError('direct-Q graph is not marked QC eligible as an oracle upper bound')
    expected_hash = 'bb5b26bf1776785d7196745d97ee1b98e177feeb7112c69ef6344c9c3b3548f9'
    if config.get('graph_hash') != expected_hash:
        raise ValueError('v0.9 graph_direct is pinned to the v0.8.4b O_np1 graph hash')
    return {'kind': 'oracle_upper_bound', 'oracle_np_max': 1, 'graph_hash': config['graph_hash']}


def _direct_action_selector(scorer, *, n_samples, action_dim, horizon, seed):
    import jax
    import jax.numpy as jnp
    from graphcritic.direct_q import puzzle_observation_window

    key = jax.random.PRNGKey(seed)
    action_size = action_dim * horizon

    def select(agent, observation, observation_history, physical_history):
        nonlocal key
        key, sample_key = jax.random.split(key)
        observations = jnp.broadcast_to(
            jnp.asarray(observation, dtype=jnp.float32), (n_samples, *np.asarray(observation).shape)
        )
        noises = jax.random.normal(sample_key, (n_samples, action_size))
        chunks = np.asarray(jax.device_get(agent.compute_flow_actions(observations, noises)))
        qpos_history = np.stack([item[0] for item in physical_history])
        button_history = np.stack([item[1] for item in physical_history])
        window, board = puzzle_observation_window(
            observation_history, scorer.state_mean, scorer.state_std,
            qpos_history=qpos_history, button_history=button_history,
            n_buttons=scorer.n_buttons, window=scorer.horizon,
        )
        if n_samples == 1:
            return chunks[0], None
        values, diagnostic = scorer.score(
            window, chunks.reshape(n_samples, horizon, action_dim), board=board,
        )
        selected = int(diagnostic['selected_index'])
        return chunks[selected], diagnostic

    return select

class LoggingHelper:
    def __init__(self, csv_loggers, wandb_logger):
        self.csv_loggers = csv_loggers
        self.wandb_logger = wandb_logger
        self.first_time = time.time()
        self.last_time = time.time()

    def log(self, data, prefix, step):
        assert prefix in self.csv_loggers, prefix
        self.csv_loggers[prefix].log(data, step=step)
        self.wandb_logger.log({f'{prefix}/{k}': v for k, v in data.items()}, step=step)

def main(_):
    target_provenance = _load_graph_target_provenance(FLAGS.graph_targets)
    direct_provenance = _load_direct_q_provenance(FLAGS.direct_q_config)
    if direct_provenance is not None:
        if (direct_provenance.get('kind') == 'oracle_upper_bound'
                and 'oracle' not in str(FLAGS.run_group).lower()):
            raise ValueError('oracle_upper_bound direct-Q runs require "oracle" in --run_group')
        if FLAGS.graph_targets is not None:
            raise ValueError('graph_direct consumes the graph config directly and must not use --graph_targets')
        target_provenance = direct_provenance
    _validate_oracle_run_group(target_provenance, FLAGS.run_group)
    FLAGS.graph_target_provenance = json.dumps(target_provenance, sort_keys=True)
    provenance_csv = _provenance_csv_value(target_provenance)
    print(f'Graph target provenance: {json.dumps(target_provenance, sort_keys=True)}', flush=True)
    exp_name = get_exp_name(FLAGS.seed)
    run = setup_wandb(project='qc', group=FLAGS.run_group, name=exp_name)
    
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, FLAGS.env_name, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()

    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    config = FLAGS.agent
    critic_mode = config.get('critic_mode', 'td')
    if critic_mode not in {'td', 'graph', 'graph_direct'}:
        raise ValueError("--agent.critic_mode must be 'td', 'graph', or 'graph_direct'")
    if config.get('graph_target_unit', 'steps') != 'steps':
        raise ValueError("QC graph-target training currently supports --agent.graph_target_unit=steps only")
    graph_target_kind = config.get('graph_target_kind', 'regret')
    if graph_target_kind not in {'cost', 'regret', 'discounted'}:
        raise ValueError("--agent.graph_target_kind must be 'cost', 'regret', or 'discounted'")
    graph_mode = critic_mode == 'graph'
    direct_q_mode = critic_mode == 'graph_direct'
    if direct_q_mode:
        if FLAGS.direct_q_config is None or direct_provenance is None:
            raise ValueError('--agent.critic_mode=graph_direct requires --direct_q_config')
        if config.get('actor_type') != 'best-of-n' or not config.get('action_chunking'):
            raise ValueError('graph_direct requires best-of-n and action chunking')
        if FLAGS.horizon_length != 5 or FLAGS.online_steps != 0:
            raise ValueError('graph_direct supports five-step chunks and offline-only training')
        if FLAGS.offline_steps != 300000 or config.get('actor_num_samples') != 32:
            raise ValueError('direct-Q runs are fixed to 300k updates and 32 policy proposals')
        if FLAGS.dataset_proportion != 1.0 or FLAGS.ogbench_dataset_dir is not None or FLAGS.ogbench_dataset_path is not None:
            raise ValueError('graph_direct is pinned to the standard OGBench dataset')
        if FLAGS.eval_bc_also:
            eval_bc_prefix = True
        else:
            eval_bc_prefix = False
    else:
        if FLAGS.direct_q_config is not None or FLAGS.eval_bc_also:
            raise ValueError('--direct_q_config and --eval_bc_also require graph_direct mode')
        eval_bc_prefix = False
    if FLAGS.ogbench_dataset_dir is not None and FLAGS.ogbench_dataset_path is not None:
        raise ValueError('choose either --ogbench_dataset_dir or --ogbench_dataset_path')
    if graph_mode and FLAGS.ogbench_dataset_dir is not None:
        raise ValueError('graph-target training does not support replaceable --ogbench_dataset_dir datasets')
    
    # data loading
    if FLAGS.ogbench_dataset_path is not None:
        env, eval_env, train_dataset, val_dataset = make_ogbench_env_and_datasets(
            FLAGS.env_name,
            dataset_path=os.path.expanduser(FLAGS.ogbench_dataset_path),
            compact_dataset=False,
        )
    elif FLAGS.ogbench_dataset_dir is not None:
        # custom ogbench dataset
        assert FLAGS.dataset_replace_interval != 0
        assert FLAGS.dataset_proportion == 1.0
        dataset_idx = 0
        dataset_paths = [
            file for file in sorted(glob.glob(f"{FLAGS.ogbench_dataset_dir}/*.npz")) if '-val.npz' not in file
        ]
        env, eval_env, train_dataset, val_dataset = make_ogbench_env_and_datasets(
            FLAGS.env_name,
            dataset_path=dataset_paths[dataset_idx],
            compact_dataset=False,
        )
    else:
        env, eval_env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name)

    direct_scorer = None
    if direct_q_mode:
        direct_cfg = json.loads(Path(FLAGS.direct_q_config).read_text())
        if FLAGS.env_name not in direct_cfg.get('dataset_id', ''):
            raise ValueError('direct-Q dataset ID does not match --env_name')
        from graphcritic.direct_q import load_direct_q_scorer
        direct_scorer = load_direct_q_scorer(FLAGS.direct_q_config)

    if FLAGS.graph_targets is not None:
        if not graph_mode:
            raise ValueError('--graph_targets requires --agent.critic_mode=graph')
        if FLAGS.online_steps > 0:
            raise ValueError('graph-target training is offline-only; set --online_steps=0')
        if FLAGS.graph_dataset_id is None:
            raise ValueError('--graph_dataset_id is required with --graph_targets to check target provenance')
        from graphcritic.io import load_targets

        targets = load_targets(
            FLAGS.graph_targets,
            expected_dataset_id=FLAGS.graph_dataset_id,
            require_loss_mask=True,
        )
        if targets.unit != config.get('graph_target_unit', 'steps'):
            raise ValueError('graph target unit does not match --agent.graph_target_unit')
        if targets.kind != graph_target_kind:
            raise ValueError(
                f'graph target kind {targets.kind!r} does not match '
                f'--agent.graph_target_kind={graph_target_kind!r}'
            )
        if targets.kind == 'discounted' and not np.isclose(
            float(targets.discount_gamma), float(config.get('graph_target_gamma', 0.99))
        ):
            raise ValueError('discounted target gamma does not match --agent.graph_target_gamma')
        if targets.loss_mask is None:
            raise ValueError('graph target artifact is missing graph_target_loss_mask')
        if targets.h != FLAGS.horizon_length:
            raise ValueError(
                f'graph target horizon {targets.h} does not match --horizon_length={FLAGS.horizon_length}'
            )
        target_dataset_id = targets.dataset_id.split(':')
        if len(target_dataset_id) < 3 or target_dataset_id[1] != FLAGS.env_name:
            raise ValueError('graph target dataset_id does not match --env_name')
        if targets.y.shape != (len(train_dataset['observations']),):
            raise ValueError('graph targets must preserve the complete flattened QC train-row count')
        expected_valid = _qc_chunk_valid(train_dataset['terminals'], FLAGS.horizon_length)
        if not np.array_equal(targets.valid, expected_valid):
            raise ValueError('graph target validity does not match QC sample_sequence boundaries')
        train_dataset = dict(train_dataset)
        train_dataset['graph_target'] = targets.y
        train_dataset['graph_target_valid'] = targets.valid
        train_dataset['graph_target_loss_mask'] = targets.loss_mask
    elif graph_mode:
        raise ValueError('--agent.critic_mode=graph requires --graph_targets')

    # house keeping
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    online_rng, rng = jax.random.split(jax.random.PRNGKey(FLAGS.seed), 2)
    log_step = 0
    
    discount = FLAGS.discount
    config["horizon_length"] = FLAGS.horizon_length

    # handle dataset
    def process_train_dataset(ds):
        """
        Process the train dataset to 
            - handle dataset proportion
            - handle sparse reward
            - convert to action chunked dataset
        """

        ds = Dataset.create(**ds)
        if FLAGS.dataset_proportion < 1.0:
            new_size = int(len(ds['masks']) * FLAGS.dataset_proportion)
            ds = Dataset.create(
                **{k: v[:new_size] for k, v in ds.items()}
            )
        
        if is_robomimic_env(FLAGS.env_name):
            penalty_rewards = ds["rewards"] - 1.0
            ds_dict = {k: v for k, v in ds.items()}
            ds_dict["rewards"] = penalty_rewards
            ds = Dataset.create(**ds_dict)
        
        if FLAGS.sparse:
            # Create a new dataset with modified rewards instead of trying to modify the frozen one
            sparse_rewards = (ds["rewards"] != 0.0) * -1.0
            ds_dict = {k: v for k, v in ds.items()}
            ds_dict["rewards"] = sparse_rewards
            ds = Dataset.create(**ds_dict)

        return ds
    
    train_dataset = process_train_dataset(train_dataset)
    example_batch = train_dataset.sample(())
    
    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    # Setup logging.
    prefixes = ["eval", "env"]
    if eval_bc_prefix:
        prefixes.append('eval_bc')
    if FLAGS.offline_steps > 0:
        prefixes.append("offline_agent")
    if FLAGS.online_steps > 0:
        prefixes.append("online_agent")

    logger = LoggingHelper(
        csv_loggers={prefix: CsvLogger(os.path.join(FLAGS.save_dir, f"{prefix}.csv")) 
                    for prefix in prefixes},
        wandb_logger=wandb,
    )

    offline_init_time = time.time()
    # Offline RL
    for i in tqdm.tqdm(range(1, FLAGS.offline_steps + 1)):
        log_step += 1

        if FLAGS.ogbench_dataset_dir is not None and FLAGS.dataset_replace_interval != 0 and i % FLAGS.dataset_replace_interval == 0:
            dataset_idx = (dataset_idx + 1) % len(dataset_paths)
            print(f"Using new dataset: {dataset_paths[dataset_idx]}", flush=True)
            train_dataset, val_dataset = make_ogbench_env_and_datasets(
                FLAGS.env_name,
                dataset_path=dataset_paths[dataset_idx],
                compact_dataset=False,
                dataset_only=True,
                cur_env=env,
            )
            train_dataset = process_train_dataset(train_dataset)

        batch = train_dataset.sample_sequence(config['batch_size'], sequence_length=FLAGS.horizon_length, discount=discount)

        agent, offline_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            offline_info['graph_target_provenance'] = provenance_csv
            logger.log(offline_info, "offline_agent", step=log_step)
        
        # saving
        if FLAGS.save_interval > 0 and i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, log_step)

        # eval
        should_eval = (
            FLAGS.eval_interval != 0 and i % FLAGS.eval_interval == 0
        ) or (
            i == FLAGS.offline_steps
            and (
                FLAGS.eval_interval == 0
                or FLAGS.offline_steps % FLAGS.eval_interval != 0
            )
        )
        if should_eval:
            # during eval, the action chunk is executed fully
            eval_kwargs = dict(
                agent=agent,
                env=eval_env,
                action_dim=example_batch["actions"].shape[-1],
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
            )
            if direct_q_mode:
                action_dim = example_batch["actions"].shape[-1]
                eval_kwargs['action_selector'] = _direct_action_selector(
                    direct_scorer, n_samples=config['actor_num_samples'],
                    action_dim=action_dim, horizon=FLAGS.horizon_length, seed=FLAGS.seed + i,
                )
            eval_info, _, _ = evaluate(**eval_kwargs)
            eval_info['graph_target_provenance'] = provenance_csv
            logger.log(eval_info, "eval", step=log_step)
            if direct_q_mode and eval_bc_prefix:
                bc_info, _, _ = evaluate(
                    **{
                        **eval_kwargs,
                        'action_selector': _direct_action_selector(
                            direct_scorer, n_samples=1,
                            action_dim=action_dim, horizon=FLAGS.horizon_length,
                            seed=FLAGS.seed + i + 0xBC,
                        ),
                    }
                )
                bc_info['graph_target_provenance'] = provenance_csv
                logger.log(bc_info, 'eval_bc', step=log_step)

    # transition from offline to online
    replay_buffer = ReplayBuffer.create_from_initial_dataset(
        dict(train_dataset), size=max(FLAGS.buffer_size, train_dataset.size + 1)
    )
        
    ob, _ = env.reset()
    
    action_queue = []
    action_dim = example_batch["actions"].shape[-1]

    # Online RL
    update_info = {}

    from collections import defaultdict
    data = defaultdict(list)
    online_init_time = time.time()
    for i in tqdm.tqdm(range(1, FLAGS.online_steps + 1)):
        log_step += 1
        online_rng, key = jax.random.split(online_rng)
        
        # during online rl, the action chunk is executed fully
        if len(action_queue) == 0:
            action = agent.sample_actions(observations=ob, rng=key)

            action_chunk = np.array(action).reshape(-1, action_dim)
            for action in action_chunk:
                action_queue.append(action)
        action = action_queue.pop(0)
        
        next_ob, int_reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if FLAGS.save_all_online_states:
            state = env.get_state()
            data["steps"].append(i)
            data["obs"].append(np.copy(next_ob))
            data["qpos"].append(np.copy(state["qpos"]))
            data["qvel"].append(np.copy(state["qvel"]))
            if "button_states" in state:
                data["button_states"].append(np.copy(state["button_states"]))
        
        # logging useful metrics from info dict
        env_info = {}
        for key, value in info.items():
            if key.startswith("distance"):
                env_info[key] = value
        # always log this at every step
        logger.log(env_info, "env", step=log_step)

        if 'antmaze' in FLAGS.env_name and (
            'diverse' in FLAGS.env_name or 'play' in FLAGS.env_name or 'umaze' in FLAGS.env_name
        ):
            # Adjust reward for D4RL antmaze.
            int_reward = int_reward - 1.0
        elif is_robomimic_env(FLAGS.env_name):
            # Adjust online (0, 1) reward for robomimic
            int_reward = int_reward - 1.0

        if FLAGS.sparse:
            assert int_reward <= 0.0
            int_reward = (int_reward != 0.0) * -1.0

        transition = dict(
            observations=ob,
            actions=action,
            rewards=int_reward,
            terminals=float(done),
            masks=1.0 - terminated,
            next_observations=next_ob,
        )
        replay_buffer.add_transition(transition)
        
        # done
        if done:
            ob, _ = env.reset()
            action_queue = []  # reset the action queue
        else:
            ob = next_ob

        if i >= FLAGS.start_training:
            batch = replay_buffer.sample_sequence(config['batch_size'] * FLAGS.utd_ratio, 
                        sequence_length=FLAGS.horizon_length, discount=discount)
            batch = jax.tree.map(lambda x: x.reshape((
                FLAGS.utd_ratio, config["batch_size"]) + x.shape[1:]), batch)

            agent, update_info["online_agent"] = agent.batch_update(batch)
            
        if i % FLAGS.log_interval == 0:
            for key, info in update_info.items():
                logger.log(info, key, step=log_step)
            update_info = {}

        if i == FLAGS.online_steps - 1 or \
            (FLAGS.eval_interval != 0 and i % FLAGS.eval_interval == 0):
            eval_info, _, _ = evaluate(
                agent=agent,
                env=eval_env,
                action_dim=action_dim,
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
            )
            eval_info['graph_target_provenance'] = provenance_csv
            logger.log(eval_info, "eval", step=log_step)

        # saving
        if FLAGS.save_interval > 0 and i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, log_step)

    end_time = time.time()

    for key, csv_logger in logger.csv_loggers.items():
        csv_logger.close()

    if FLAGS.save_all_online_states:
        c_data = {"steps": np.array(data["steps"]),
                 "qpos": np.stack(data["qpos"], axis=0), 
                 "qvel": np.stack(data["qvel"], axis=0), 
                 "obs": np.stack(data["obs"], axis=0), 
                 "offline_time": online_init_time - offline_init_time,
                 "online_time": end_time - online_init_time,
        }
        if len(data["button_states"]) != 0:
            c_data["button_states"] = np.stack(data["button_states"], axis=0)
        np.savez(os.path.join(FLAGS.save_dir, "data.npz"), **c_data)

    # Offline W&B runs have no URL. Keep the run artifacts local without
    # turning an otherwise completed training job into a shutdown error.
    if run.url is not None:
        with open(os.path.join(FLAGS.save_dir, 'token.tk'), 'w') as f:
            f.write(run.url)

if __name__ == '__main__':
    app.run(main)
