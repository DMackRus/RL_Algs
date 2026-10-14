import torch as T
import numpy as np
import os
import time
import yaml
from collections import defaultdict

from replay_buffer import ReplayBuffer, Episode
from env import make_env
from lewm import LeWM
from goals import goal_observation


# Goal-conditioned evaluation results, in logging order.
EVAL_KEYS = ["eval_reward", "goal_dist_best", "goal_dist_final", "goal_success_rate"]


class Logger:
    """Collects per-iteration training / evaluation results and update metrics.

    One "iteration" is one pass of the outer loop in ``testing_run``: collect a
    single episode, then run a batch of model updates. Evaluation is only run
    every few iterations (``eval_policy_every``); on the iterations in between
    we repeat the most recent eval results so the eval arrays line up 1:1 with
    the training arrays and can be plotted against a shared x-axis without any
    bookkeeping.

    LeWM is reward-free, so evaluation is scored by distance to a goal
    observation (see goals.py). The env reward is still logged for reference.

    Outputs (both written to ``work_dir``):
      - ``log.txt``                   human-readable, one row per iteration
      - ``training_stats_{seed}.npz`` all history as numpy arrays for offline plotting
    """

    def __init__(self, work_dir, cfg, seed):
        self.work_dir = work_dir
        self.cfg = cfg
        self.seed = seed
        os.makedirs(work_dir, exist_ok=True)
        self.log_file = os.path.join(work_dir, "log.txt")

        # Per-iteration history (all lists stay the same length).
        self.iterations = []
        self.steps = []
        self.elapsed = []
        self.train_rewards = []
        self.train_lengths = []
        self.evals = {k: [] for k in EVAL_KEYS}
        self.metrics = defaultdict(list)   # update-metric name -> per-iteration value

        # Most recent evaluation result, carried forward between eval runs.
        self._last_eval = {k: float("nan") for k in EVAL_KEYS}

        self.start_time = time.time()

    def set_eval(self, results):
        """Record a fresh evaluation result (dict keyed by EVAL_KEYS). Call on eval
        iterations only; the values are repeated until the next call."""
        self._last_eval = {k: float(results[k]) for k in EVAL_KEYS}

    def log_iteration(self, iteration, step, episode, train_metrics):
        """Store everything for one outer training-loop iteration."""
        n = len(self.iterations)

        self.iterations.append(int(iteration))
        self.steps.append(int(step))
        self.elapsed.append(time.time() - self.start_time)
        self.train_rewards.append(float(episode.cumulative_reward))
        self.train_lengths.append(int(len(episode)))

        # Repeat the latest eval results so eval arrays align with training arrays.
        for k in EVAL_KEYS:
            self.evals[k].append(self._last_eval[k])

        # Update metrics. ``train_metrics`` is empty during the seed phase, and a
        # metric first seen mid-run is back-filled with NaN for earlier rows.
        for key in set(self.metrics) | set(train_metrics):
            col = self.metrics[key]
            col.extend([float("nan")] * (n - len(col)))
            col.append(float(train_metrics.get(key, float("nan"))))

        self._write_log()

    def _write_log(self):
        metric_keys = sorted(self.metrics)
        cols = ["iter", "step", "elapsed_s", "train_reward", "train_length"] + EVAL_KEYS + metric_keys
        lines = ["\t".join(cols)]
        for i in range(len(self.iterations)):
            row = [self.iterations[i], self.steps[i], f"{self.elapsed[i]:.1f}",
                   f"{self.train_rewards[i]:.2f}", self.train_lengths[i]]
            row += [f"{self.evals[k][i]:.3f}" for k in EVAL_KEYS]
            row += [f"{self.metrics[k][i]:.4f}" for k in metric_keys]
            lines.append("\t".join(str(x) for x in row))
        with open(self.log_file, "w") as f:
            f.write("\n".join(lines) + "\n")

    def save(self, filename="training_stats"):
        """Dump all collected history to a ``.npz`` for offline plotting."""
        arrays = dict(
            iterations=np.array(self.iterations),
            steps=np.array(self.steps),
            elapsed_seconds=np.array(self.elapsed),
            training_episode_rewards=np.array(self.train_rewards),
            training_episode_lengths=np.array(self.train_lengths),
            evaluation_episode_rewards=np.array(self.evals["eval_reward"]),
            evaluation_goal_dist_best=np.array(self.evals["goal_dist_best"]),
            evaluation_goal_dist_final=np.array(self.evals["goal_dist_final"]),
            evaluation_goal_success_rate=np.array(self.evals["goal_success_rate"]),
        )
        for key, values in self.metrics.items():
            arrays[f"metric_{key}"] = np.array(values)
        np.savez(os.path.join(self.work_dir, f"{filename}_{self.seed}.npz"), **arrays)


@T.no_grad()
def evaluate(env, agent, num_episodes, step, cfg):
    """Goal-conditioned rollouts. Reward-free: score each episode by how close
    the achieved observation gets to a task-specific goal observation (see
    goals.py). Success == best distance < cfg['goal_success_dist']. Returns the
    episode means, keyed by EVAL_KEYS."""
    thr = float(cfg.get("goal_success_dist", 0.1))
    rewards, finals, bests, succ = [], [], [], []
    for _ in range(num_episodes):
        obs = env.reset()
        goal = goal_observation(env)
        agent.set_goal(goal)
        done, t, ep_reward = False, 0, 0.0
        d, best = float("inf"), float("inf")
        while not done:
            action = agent.plan(obs, eval_mode=True, step=step, t0=(t == 0))
            obs, reward, done, _ = env.step(action.cpu().numpy())
            ep_reward += reward
            d = float(np.linalg.norm(np.asarray(obs, np.float32) - goal))
            best = min(best, d)
            t += 1
        rewards.append(ep_reward)
        finals.append(d)
        bests.append(best)
        succ.append(float(best < thr))
    return dict(
        eval_reward=float(np.mean(rewards)),
        goal_dist_best=float(np.mean(bests)),
        goal_dist_final=float(np.mean(finals)),
        goal_success_rate=float(np.mean(succ)),
    )


def set_seed(seed):
    """Set the random seed for reproducibility."""
    np.random.seed(seed)
    T.manual_seed(seed)
    if T.cuda.is_available():
        T.cuda.manual_seed_all(seed)


def testing_run(config_filepath):

    # Load the config yaml file
    folder_path = os.path.dirname(config_filepath)
    with open(config_filepath, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    EVAL_EVERY = config["eval_policy_every"]          # eval once every N iterations
    EVAL_EPISODES = config.get("eval_episodes", 5)    # rollouts averaged per eval

    # Create the environment (benchmark / task / action_repeat set in the config).
    env = make_env(config)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    config["action_dim"] = action_dim
    config["state_dim"] = state_dim
    config["episode_length"] = env.ep_len

    print(f"Env control timestep: {env.control_timestep()}")
    print(f"Env episode length: {config['episode_length']}")

    print("Training starts...")

    for seed in range(config.get("num_seeds", 1)):
        set_seed(seed)
        episode_idx = 0
        replay_buffer = ReplayBuffer(config)
        logger = Logger(folder_path, config, seed=seed)
        agent = LeWM(config)

        for step in range(0, config["train_steps"]+config["episode_length"], config["episode_length"]):

            # Plan towards a goal during collection too (random actions until seed_steps).
            obs = env.reset()
            agent.set_goal(goal_observation(env))
            episode = Episode(config, obs)
            while not episode.done:
                action = agent.plan(obs, step=step, t0=episode.first)
                obs, reward, done, _ = env.step(action.cpu().numpy())
                episode += (obs, action, reward, done)
            assert len(episode) == config["episode_length"]
            replay_buffer += episode

            print(f"Step {step}: Collected episode with reward {episode.cumulative_reward:.2f} and length {len(episode)}")

            # Update the model
            train_metrics = {}
            if step >= config["seed_steps"]:
                num_updates = config["seed_steps"] if step == config["seed_steps"] else config["episode_length"]
                for i in range(num_updates):
                    train_metrics.update(agent.update(replay_buffer, step+i))

            # Evaluate periodically (noise-free goal-conditioned planning). The
            # results are carried forward by the logger onto the intervening iterations.
            if step >= config["seed_steps"] and episode_idx % EVAL_EVERY == 0:
                results = evaluate(env, agent, EVAL_EPISODES, step, config)
                logger.set_eval(results)
                print(f"Step {step}: Eval over {EVAL_EPISODES} episodes: "
                      f"goal dist (best) {results['goal_dist_best']:.3f}, "
                      f"final {results['goal_dist_final']:.3f}, "
                      f"success {results['goal_success_rate']:.2f}, "
                      f"env reward {results['eval_reward']:.2f}")

                os.makedirs(f"{folder_path}/model_checkpoints", exist_ok=True)
                agent.save(f"{folder_path}/model_checkpoints/checkpoint_step{step}.pt")

            # Record this iteration: training reward/length, carried-forward eval
            # results, and the latest update metrics (NaN during seed phase).
            logger.log_iteration(episode_idx, step, episode, train_metrics)
            logger.save()

            episode_idx += 1

        logger.save()


if __name__ == "__main__":

    testing_run("configs/reacher_hard/config.yaml")
