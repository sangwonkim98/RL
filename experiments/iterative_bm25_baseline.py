"""
Iterative BM25 baseline (no RL):
  매 step: 미독 docs 중 BM25 score 최고 doc 읽음
  LLM sufficiency=1 되면 answer 생성
  max_steps 도달하면 강제 answer

이게 PPO와 거의 동일한 행동 패턴 → "PPO가 RL로 추가 학습한 게 있나?" 검증용
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from envs.doc_reader_env import (
    DocReaderEnv, N_DOCS, ANSWER_ACTION, tokenize, BM25Lite, answer_correct
)


def iterative_bm25_rollout(env, ep):
    """Greedy: 미독 docs 중 BM25 top-1 읽음. LLM suff=1 되면 answer."""
    env._init_episode(ep)
    obs = env._obs()
    total_r = 0.0
    info = {}
    done = False
    while not done:
        # 1) LLM suff=1이면 answer
        if env.llm_sufficient == 1:
            a = ANSWER_ACTION
        else:
            # 2) 미독 중 BM25 최고
            unread_scores = np.where(
                np.array(env.read_mask) == 0, env.bm25_scores, -1.0
            )
            if unread_scores.max() < 0:
                # 다 읽음 → answer 강제
                a = ANSWER_ACTION
            else:
                a = int(np.argmax(unread_scores))
        obs, r, term, trunc, info = env.step(a)
        total_r += r
        if term or trunc:
            done = True
    return total_r, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/hotpot_doc_reader.json")
    ap.add_argument("--max-steps", type=int, default=4)
    ap.add_argument("--n", type=int, default=199)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--vllm-url", default="http://localhost:8002/v1")
    ap.add_argument("--vllm-model", default="Qwen/Qwen2.5-7B-Instruct")
    args = ap.parse_args()

    with open(args.data) as f:
        episodes = json.load(f)
    episodes = episodes[: args.n]

    env = DocReaderEnv(
        args.data,
        max_steps=args.max_steps,
        vllm_url=args.vllm_url,
        vllm_model=args.vllm_model,
        dry_run=args.dry_run,
    )

    successes, returns, lengths, gold_hits, reads_count = [], [], [], [], []
    for i, ep in enumerate(episodes):
        r, info = iterative_bm25_rollout(env, ep)
        successes.append(float(info.get("success", False)))
        returns.append(r)
        lengths.append(info["step"])
        gh = sum(1 for idx in ep["gold_indices"] if env.read_mask[idx])
        gold_hits.append(gh)
        reads_count.append(sum(env.read_mask))
        if (i + 1) % 50 == 0:
            print(f"  ...{i+1}/{len(episodes)} done")

    print(f"\n=== Iterative BM25 baseline (n={len(episodes)}, max_steps={args.max_steps}, dry_run={args.dry_run}) ===")
    print(f"  success_rate : {np.mean(successes):.2%}")
    print(f"  avg_return   : {np.mean(returns):+.3f}")
    print(f"  avg_length   : {np.mean(lengths):.2f}")
    print(f"  avg_reads    : {np.mean(reads_count):.2f}")
    print(f"  avg_gold_hit : {np.mean(gold_hits):.2f} / 2")


if __name__ == "__main__":
    main()
