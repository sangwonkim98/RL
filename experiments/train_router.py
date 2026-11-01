import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import PPO

from envs.path_router_env import PathRouterEnv


def evaluate(model, env, n_episodes=300):
    returns, successes, lengths = [], [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_return, ep_len, success = 0.0, 0, False
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
    ap.add_argument("--data", default="data/hotpot_path_data.json")
    ap.add_argument("--timesteps", type=int, default=200000)
    ap.add_argument("--out", default="logs/ppo_router.zip")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path("logs/tb_router").mkdir(parents=True, exist_ok=True)

    env = PathRouterEnv(args.data)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        tensorboard_log="logs/tb_router/",
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        learning_rate=3e-4,
    )

    print(f"Training PPO Router  |  timesteps={args.timesteps}  |  data={args.data}")
    model.learn(total_timesteps=args.timesteps)
    model.save(args.out)
    print(f"Saved → {args.out}")

    print("\nEvaluating (300 episodes, deterministic)...")
    avg_ret, succ_rate, avg_len = evaluate(model, env)
    print(f"avg_return   : {avg_ret:.3f}")
    print(f"success_rate : {succ_rate:.2%}")
    print(f"avg_length   : {avg_len:.2f}")


if __name__ == "__main__":
    main()
