"""
학습된 PPO Router 평가 (rollout 출력 + 전체 통계)
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import PPO

from envs.doc_reader_env import DocReaderEnv, N_DOCS, ANSWER_ACTION


def rollout(model, env, ep):
    env._init_episode(ep)
    obs = env._obs()

    traj, total_r = [], 0.0
    info = {}
    done = False
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        a = int(a)
        obs, r, term, trunc, info = env.step(a)
        if a == ANSWER_ACTION:
            traj.append(
                f"answer → '{info.get('answer','?')[:40]}' "
                f"(correct={info.get('correct')})  r={r:+.2f}"
            )
        else:
            mark = "★" if a in ep["gold_indices"] else " "
            traj.append(
                f"read doc#{a} {mark} [{ep['docs'][a]['title'][:40]}]  r={r:+.2f}"
            )
        total_r += r
        if term or trunc:
            done = True
    return traj, total_r, info.get("success", False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="logs/ppo_doc_reader.zip")
    ap.add_argument("--data", default="data/hotpot_doc_reader.json")
    ap.add_argument("--n", type=int, default=10, help="rollout 출력할 episode 수")
    ap.add_argument("--eval", type=int, default=100, help="통계용 episode 수")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-steps", type=int, default=4)
    ap.add_argument("--vllm-url", default="http://localhost:8002/v1")
    ap.add_argument("--vllm-model", default="Qwen/Qwen2.5-7B-Instruct")
    args = ap.parse_args()

    with open(args.data) as f:
        episodes = json.load(f)

    env = DocReaderEnv(
        args.data,
        max_steps=args.max_steps,
        vllm_url=args.vllm_url,
        vllm_model=args.vllm_model,
        dry_run=args.dry_run,
    )
    model = PPO.load(args.model, env=env)

    # ── 샘플 rollout ──
    print("\n=== Sample Rollouts ===")
    for ep in episodes[: args.n]:
        traj, total_r, success = rollout(model, env, ep)
        flag = "✓" if success else "✗"
        print(f"\n[{flag}] {ep['question'][:80]}")
        print(f"     gold_idx={ep['gold_indices']}  return={total_r:+.2f}")
        for line in traj:
            print(f"     {line}")

    # ── 통계 ──
    print(f"\n=== Stats over {args.eval} episodes ===")
    random.seed(42)
    sample = random.sample(episodes, min(args.eval, len(episodes)))
    succ, rets, lens, gold_hits = [], [], [], []
    for ep in sample:
        traj, r, s = rollout(model, env, ep)
        succ.append(float(s))
        rets.append(r)
        lens.append(len(traj))
        gh = sum(1 for ln in traj if ln.startswith("read") and "★" in ln)
        gold_hits.append(gh)
    print(f"  success_rate : {np.mean(succ):.2%}")
    print(f"  avg_return   : {np.mean(rets):+.3f}")
    print(f"  avg_length   : {np.mean(lens):.2f}")
    print(f"  avg_gold_hit : {np.mean(gold_hits):.2f} / 2")


if __name__ == "__main__":
    main()
