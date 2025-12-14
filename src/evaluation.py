import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env

from rsaenv import RSAEnv

MOVING_AVG = 10


def moving_avg(x, w):
    x = np.array(x, dtype=float)
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def evaluate_agent(capacity: int, model_path: str, num_episodes: int):
    """
    Run deterministic evaluation on data/eval and return per-episode blocking rates.
    """

    print(f"\n=== EVALUATING DQN FOR CAPACITY = {capacity} ===")
    print(f"Model: {model_path}")

    # We are using the same environment as in training, but with data from a set eval folder
    env = make_vec_env(
        lambda: RSAEnv(capacity=capacity, data_dir="../data/eval"),
        n_envs=1
    )

    model = DQN.load(model_path, env=env)

    episode_blocking_rates = []

    for ep in range(num_episodes):
        obs = env.reset()
        done = False

        blocked = 0
        total = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = env.step(action)

            info = infos[0]
            if "blocked" in info:
                blocked += int(info["blocked"])
                total += 1

        br = blocked / total if total > 0 else 0.0
        episode_blocking_rates.append(br)

        print(f"Episode {ep + 1}/{num_episodes} | Blocking rate = {br:.3f}")

    return episode_blocking_rates

# this time we plot average blocking rate vs episode
def plot_curve(values, title, ylabel, save_path):
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

    print(f"Saved evaluation plot to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capacity", type=int, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=2000)

    args = parser.parse_args()

    capacity=args.capacity

    blocking_rates = evaluate_agent(
        capacity,
        model_path=args.model_path,
        num_episodes=args.episodes
    )

    save_path = f"../plots/eval_blocking_rate_capacity_{args.capacity}.png"
    plot_curve(blocking_rates,
        f"Evaluation Objective vs Episode (capacity={capacity})",
        "Average Objective B (Blocking Rate)", save_path)


if __name__ == "__main__":
    main()
