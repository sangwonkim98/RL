"""
PPO training on DocReaderEnv (LLM in loop)

Usage:
  # dry-run (LLM 없이 oracle 신호, ~5분)
  python -m experiments.train_doc_reader --dry-run --timesteps 10000

  # 32B real LLM
  python -m experiments.train_doc_reader \
      --timesteps 15000 \
      --vllm-url http://localhost:8003/v1 \
      --vllm-model Qwen/Qwen2.5-32B-Instruct-AWQ
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from envs.doc_reader_env import DocReaderEnv


class ProgressCallback(BaseCallback):
    def __init__(self, log_every=1000):
        super().__init__()
        self.log_every = log_every
        self.start = None

    def _on_training_start(self):
        self.start = time.time()

    def _on_step(self):
        if self.num_timesteps % self.log_every == 0:
            elapsed = time.time() - self.start
            rate = self.num_timesteps / max(elapsed, 1e-3)
            print(f"[step {self.num_timesteps}] elapsed={elapsed:.1f}s  "
                  f"rate={rate:.1f}/s")
        return True


def evaluate(model, env, n=50):
    returns, succ, lens, n_reads, n_gold = [], [], [], [], []
    for _ in range(n):
        obs, _ = env.reset()
        done = False
        ep_r, ep_l, s = 0.0, 0, False
        gold_idx = set(env.ep["gold_indices"])
        reads = 0
        golds_hit = 0
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            a = int(a)
            if a < 10:
                reads += 1
                if a in gold_idx:
                    golds_hit += 1
            obs, r, term, trunc, info = env.step(a)
            ep_r += r
            ep_l += 1
            if term or trunc:
                done = True
                s = info.get("success", False)
        returns.append(ep_r)
        succ.append(float(s))
        lens.append(ep_l)
        n_reads.append(reads)
        n_gold.append(golds_hit)
    return {
        "avg_return": float(np.mean(returns)),
        "success_rate": float(np.mean(succ)),
        "avg_length": float(np.mean(lens)),
        "avg_reads": float(np.mean(n_reads)),
        "avg_gold_hit": float(np.mean(n_gold)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/hotpot_doc_reader.json")
    ap.add_argument("--timesteps", type=int, default=15000)
    ap.add_argument("--out", default="logs/ppo_doc_reader.zip")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--vllm-url", default="http://localhost:8002/v1")
    ap.add_argument("--vllm-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--max-steps", type=int, default=6)
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path("logs/tb_doc").mkdir(parents=True, exist_ok=True)

    cache = {}
    env = DocReaderEnv(
        args.data,
        max_steps=args.max_steps,
        vllm_url=args.vllm_url,
        vllm_model=args.vllm_model,
        llm_cache=cache,
        dry_run=args.dry_run,
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        tensorboard_log="logs/tb_doc/",
        n_steps=1024,
        batch_size=128,
        n_epochs=8,
        learning_rate=5e-4,
        gamma=0.99,
        ent_coef=0.03,
    )

    print(f"Training DocReader Router")
    print(f"  data       : {args.data} ({len(env.episodes)} episodes)")
    print(f"  timesteps  : {args.timesteps}")
    print(f"  dry_run    : {args.dry_run}")
    if not args.dry_run:
        print(f"  vllm_url   : {args.vllm_url}")
        print(f"  vllm_model : {args.vllm_model}")
    print()

    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=ProgressCallback(log_every=512))
    train_time = time.time() - t0
    model.save(args.out)
    print(f"\nSaved → {args.out}")
    print(f"Training time: {train_time:.1f}s")
    print(f"LLM cache size: {len(cache)}")

    print("\nEvaluating (50 episodes, deterministic)...")
    metrics = evaluate(model, env, n=50)
    for k, v in metrics.items():
        print(f"  {k:15s}: {v:.3f}")


if __name__ == "__main__":
    main()
