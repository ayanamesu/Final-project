# dqn_runner.py
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

from rsaenv import RSAEnv

# Window size for moving average (assignment says last 10 episodes)
MOVING_AVG = 10


def moving_avg(x, w):
    """
    Simple moving average over window w.
    If there are fewer than w points, just return the original list as a numpy array.
    """
    x = np.array(x, dtype=float)
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


class TrainCallback(BaseCallback):
    """
    Collects per-episode reward and blocking rate during training.

    Assumes:
    - Single environment (n_envs=1)
    - info["blocked"] is 0 if the current request was served, 1 if it was blocked
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.ep_rewards = []        # list of episode total rewards
        self.ep_blocking_rates = [] # list of episode blocking rates

        # internal counters for the current episode
        self._curr_reward = 0.0
        self._curr_blocked = 0
        self._curr_requests = 0

    def _on_step(self) -> bool:
        # SB3 passes a dict in self.locals containing "rewards", "dones", "infos", etc.
        rewards = self.locals["rewards"][0]  # single env
        dones = self.locals["dones"][0]
        infos = self.locals["infos"][0]

        # accumulate reward
        self._curr_reward += float(rewards)

        # accumulate blocking stats (per-request)
        if "blocked" in infos:
            self._curr_blocked += int(infos["blocked"])
            self._curr_requests += 1

        # if episode finished, store stats and reset counters
        if dones:
            self.ep_rewards.append(self._curr_reward)

            if self._curr_requests > 0:
                br = self._curr_blocked / self._curr_requests
            else:
                br = 0.0
            self.ep_blocking_rates.append(br)

            # reset episode counters
            self._curr_reward = 0.0
            self._curr_blocked = 0
            self._curr_requests = 0

        return True


def plot_curve(values, title, ylabel, save_path):
    """
    Plot a moving-averaged curve and save to file.
    """
    ma = moving_avg(values, MOVING_AVG)

    plt.figure()
    plt.plot(ma)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def train_agent(capacity: int):
    """
    Train a DQN agent for a given link capacity.

    - Uses data from ../data/train
    - Saves model to ../models
    - Saves training plots to ../plots
    """

    print(f"\n=== TRAINING DQN FOR CAPACITY = {capacity} ===")

    # Create vectorized env (required by SB3)
    # We point data_dir to ../data/train because dqn_runner.py is in src/
    env = make_vec_env(
        lambda: RSAEnv(capacity=capacity, data_dir="../data/train"),
        n_envs=1
    )

    callback = TrainCallback()

    # DQN hyperparameters (you can mention/tune these in README)
    model = DQN(
        "MlpPolicy",        # observation is a Box, so MlpPolicy is correct
        env,
        learning_rate=1e-3,
        batch_size=64,
        buffer_size=50000,
        learning_starts=1000,
        target_update_interval=1000,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        gamma=0.99,
        verbose=1,
    )

    # Total timesteps: adjust if you want longer training
    # 200_000 is a reasonable starting point
    model.learn(
        total_timesteps=200_000,
        callback=callback,
        log_interval=10
    )
    print("Episodes recorded:", len(callback.ep_rewards), len(callback.ep_blocking_rates))
    
    # Save model
    models_dir = "../models"
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"dqn_capacity_{capacity}.zip")
    model.save(model_path)
    print(f"Saved model to {model_path}")

    # Save training plots
    plots_dir = "../plots"

    plot_curve(
        callback.ep_rewards,
        f"Learning Curve (capacity={capacity})",
        "Avg episode reward (moving avg)",
        os.path.join(plots_dir, f"learning_curve_capacity_{capacity}.png")
    )

    plot_curve(
        callback.ep_blocking_rates,
        f"Training Blocking Rate (capacity={capacity})",
        "Blocking rate (moving avg)",
        os.path.join(plots_dir, f"training_blocking_rate_capacity_{capacity}.png")
    )

    print(f"Saved training plots to {plots_dir}")

    return model_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--capacity",
        type=int,
        required=True,
        help="Link capacity (e.g., 20 for Part 1, 10 for Part 2)",
    )
    args = parser.parse_args()

    train_agent(args.capacity)


if __name__ == "__main__":
    main()
