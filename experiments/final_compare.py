"""
최종 비교: PPO (real 7B 학습) vs Iterative BM25 baseline (real 7B 평가)
3가지 metric: success / gold_hit / read 횟수
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import PPO
from envs.doc_reader_env import DocReaderEnv, N_DOCS, ANSWER_ACTION


def ppo_rollout(model, env, ep):
    env._init_episode(ep)
    obs = env._obs()
    total_r = 0.0
    info = {}
    done = False
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(int(a))
        total_r += r
        if term or trunc:
            done = True
    return info


def iterative_bm25_rollout(env, ep):
    """매 step: 미독 BM25 top-1 read. LLM suff=1 또는 max_steps 되면 answer."""
    env._init_episode(ep)
    obs = env._obs()
    info = {}
    done = False
    while not done:
        if env.llm_sufficient == 1:
            a = ANSWER_ACTION
        else:
            unread_scores = np.where(
                np.array(env.read_mask) == 0, env.bm25_scores, -1.0
            )
            if unread_scores.max() < 0:
                a = ANSWER_ACTION
            else:
                a = int(np.argmax(unread_scores))
        obs, r, term, trunc, info = env.step(a)
        if term or trunc:
            done = True
    return info


def collect_metrics(env, episodes, rollout_fn, label):
    print(f"\n=== {label} ===")
    succ, lens, golds, reads = [], [], [], []
    for i, ep in enumerate(episodes):
        info = rollout_fn(ep)
        succ.append(float(info.get("success", False)))
        lens.append(info["step"])
        gh = sum(1 for idx in ep["gold_indices"] if env.read_mask[idx])
        golds.append(gh)
        reads.append(sum(env.read_mask))
        if (i + 1) % 50 == 0:
            print(f"  ...{i+1}/{len(episodes)}  succ_so_far={np.mean(succ):.2%}")
    return {
        "label": label,
        "n": len(episodes),
        "success_rate": float(np.mean(succ)),
        "avg_gold_hit": float(np.mean(golds)),
        "avg_reads":    float(np.mean(reads)),
        "avg_length":   float(np.mean(lens)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppo-model", default="logs/ppo_real_7b_100k.zip")
    ap.add_argument("--data", default="data/hotpot_doc_reader.json")
    ap.add_argument("--max-steps", type=int, default=4)
    ap.add_argument("--vllm-url", default="http://localhost:8002/v1")
    ap.add_argument("--vllm-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--n", type=int, default=199)
    ap.add_argument("--out", default="logs/final_compare.json")
    args = ap.parse_args()

    with open(args.data) as f:
        episodes = json.load(f)
    episodes = episodes[: args.n]

    # 캐시 공유 (둘이 같은 LLM 호출 결과 재사용)
    shared_cache = {}
    env = DocReaderEnv(
        args.data,
        max_steps=args.max_steps,
        vllm_url=args.vllm_url,
        vllm_model=args.vllm_model,
        llm_cache=shared_cache,
        dry_run=False,    # ← real LLM
    )

    print(f"PPO model: {args.ppo_model}")
    print(f"vLLM     : {args.vllm_model} @ {args.vllm_url}")
    print(f"Episodes : {len(episodes)}")
    print(f"max_steps: {args.max_steps}")

    # 1) Iterative BM25 (먼저 — PPO 평가 시 캐시 재사용 가능)
    m_bm25 = collect_metrics(
        env, episodes,
        lambda ep: iterative_bm25_rollout(env, ep),
        "Iterative BM25 (real 7B)",
    )

    # 2) PPO
    model = PPO.load(args.ppo_model, env=env)
    m_ppo = collect_metrics(
        env, episodes,
        lambda ep: ppo_rollout(model, env, ep),
        "PPO Router (real 7B trained, real 7B eval)",
    )

    # 결과 print
    print("\n" + "=" * 70)
    print(f"{'Method':<48} {'Success':>9} {'GoldHit':>9} {'Reads':>7}")
    print("-" * 70)
    for m in [m_bm25, m_ppo]:
        print(f"{m['label']:<48} "
              f"{m['success_rate']:>8.2%} "
              f"{m['avg_gold_hit']:>6.2f}/2 "
              f"{m['avg_reads']:>7.2f}")
    print("=" * 70)

    # 차이
    print(f"\nPPO − Iterative BM25:")
    print(f"  Δ success   : {(m_ppo['success_rate']-m_bm25['success_rate'])*100:+.1f}%p")
    print(f"  Δ gold_hit  : {m_ppo['avg_gold_hit']-m_bm25['avg_gold_hit']:+.2f}")
    print(f"  Δ reads     : {m_ppo['avg_reads']-m_bm25['avg_reads']:+.2f}")

    # 저장
    out = {"iterative_bm25": m_bm25, "ppo_router": m_ppo,
           "ppo_model": args.ppo_model, "vllm_model": args.vllm_model}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {args.out}")
    print(f"Final cache size: {len(shared_cache)}")


if __name__ == "__main__":
    main()
