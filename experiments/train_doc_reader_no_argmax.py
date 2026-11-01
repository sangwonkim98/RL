"""PPO training without bm25_argmax_unread feature"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import PPO

from envs.doc_reader_env_no_argmax import DocReaderEnvNoArgmax


def evaluate(model, env, n=100):
    succ, rets, lens, gh = [], [], [], []
    for _ in range(n):
        obs, _ = env.reset()
        done = False
        r, l, s, gold_hit = 0, 0, False, 0
        gold_idx = set(env.ep["gold_indices"])
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            a = int(a)
            if a < 10 and a in gold_idx and env.read_mask[a] == 0:
                gold_hit += 1
            obs, rr, term, trunc, info = env.step(a)
            r += rr
            l += 1
            if term or trunc:
                done = True
                s = info.get("success", False)
        succ.append(float(s))
        rets.append(r)
        lens.append(l)
        gh.append(gold_hit)
    return {
        "success_rate": float(np.mean(succ)),
        "avg_return": float(np.mean(rets)),
        "avg_length": float(np.mean(lens)),
        "avg_gold_hit": float(np.mean(gh)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/hotpot_doc_reader.json")
    ap.add_argument("--timesteps", type=int, default=100000)
    ap.add_argument("--out", default="logs/ppo_doc_reader_no_argmax.zip")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-steps", type=int, default=4)
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path("logs/tb_doc").mkdir(parents=True, exist_ok=True)

    env = DocReaderEnvNoArgmax(args.data, max_steps=args.max_steps, dry_run=args.dry_run)
    model = PPO(
        "MlpPolicy", env, verbose=0,
        tensorboard_log="logs/tb_doc/",
        n_steps=1024, batch_size=128, n_epochs=8,
        learning_rate=5e-4, ent_coef=0.03, gamma=0.99,
    )

    t0 = time.time()
    print(f"Training no-argmax PPO  | timesteps={args.timesteps}  | dry={args.dry_run}")
    model.learn(total_timesteps=args.timesteps)
    model.save(args.out)
    print(f"Train time: {time.time()-t0:.1f}s")

    metrics = evaluate(model, env, n=100)
    print("\nEval (100 episodes):")
    for k, v in metrics.items():
        print(f"  {k:15s}: {v:.3f}")


if __name__ == "__main__":
    main()
