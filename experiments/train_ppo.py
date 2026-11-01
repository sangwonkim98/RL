import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import PPO

from envs.multihop_nav_env import MultihopNavEnv


def evaluate(model, env, n_episodes=100):
    returns, successes, lengths = [], [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        ep_len = 0
        success = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(int(action))
            ep_return += r
            ep_len += 1
            if term or trunc:
                done = True
                success = info.get("success", False)
        returns.append(ep_return)
        successes.append(float(success))
        lengths.append(ep_len)
    return float(np.mean(returns)), float(np.mean(successes)), float(np.mean(lengths))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/level1_fixed.json")
    ap.add_argument("--timesteps", type=int, default=50000)
    ap.add_argument("--out", default="logs/ppo_level1.zip")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path("logs/tb").mkdir(parents=True, exist_ok=True)

    env = MultihopNavEnv(args.data)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        tensorboard_log="logs/tb/",
    )

    print(f"Training PPO for {args.timesteps} timesteps...")
    model.learn(total_timesteps=args.timesteps)
    model.save(args.out)
    print(f"Model saved to {args.out}")

    print("\nEvaluating on 100 episodes...")
    avg_ret, succ_rate, avg_len = evaluate(model, env, n_episodes=100)
    print(f"avg_return:   {avg_ret:.3f}")
    print(f"success_rate: {succ_rate:.2%}")
    print(f"avg_length:   {avg_len:.2f}")


if __name__ == "__main__":
    main()
