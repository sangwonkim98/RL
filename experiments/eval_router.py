"""
학습된 PPO Router로 episode rollout + success rate 출력
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import PPO

from envs.path_router_env import PathRouterEnv, STOP

ACTION_NAMES = {0: "retrieve_seed", 1: "retrieve_bridge", 2: "retrieve_relation", 3: "stop"}


def rollout(model, env, ep):
    env.ep = ep
    env._init_episode(ep)
    obs = env._obs()

    trajectory, total_reward = [], 0.0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        obs, r, term, trunc, info = env.step(action)
        trajectory.append({
            "step": info["step"],
            "action": ACTION_NAMES[action],
            "gold": ACTION_NAMES[info["gold_action"]],
            "reward": round(r, 3),
            "hop_status": info["hop_status"],
        })
        total_reward += r
        if term or trunc:
            done = True
    return trajectory, round(total_reward, 3), info.get("success", False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="logs/ppo_router.zip")
    ap.add_argument("--data", default="data/hotpot_path_data.json")
    ap.add_argument("--n", type=int, default=10, help="episodes to print")
    ap.add_argument("--eval", type=int, default=500, help="total episodes for stats")
    args = ap.parse_args()

    with open(args.data) as f:
        episodes = json.load(f)

    env = PathRouterEnv(args.data)
    model = PPO.load(args.model, env=env)

    # ── 개별 롤아웃 출력 ──────────────────────────────────────────────
    print("\n=== Sample Rollouts ===")
    for ep in episodes[: args.n]:
        traj, total_r, success = rollout(model, env, ep)
        mark = "✓" if success else "✗"
        print(f"\n[{mark}] {ep['question'][:70]}")
        print(f"     return={total_r}  gold={ep['gold_actions']}")
        for t in traj:
            ok = "✓" if t["action"] == t["gold"] else "✗"
            print(f"     step{t['step']}: {t['action']:20s} {ok}  r={t['reward']}")

    # ── 전체 통계 ─────────────────────────────────────────────────────
    print(f"\n=== Stats over {args.eval} episodes ===")
    import random
    successes, returns = [], []
    sample = random.sample(episodes, min(args.eval, len(episodes)))
    for ep in sample:
        _, total_r, success = rollout(model, env, ep)
        successes.append(float(success))
        returns.append(total_r)
    print(f"success_rate : {np.mean(successes):.2%}")
    print(f"avg_return   : {np.mean(returns):.3f}")


if __name__ == "__main__":
    main()
