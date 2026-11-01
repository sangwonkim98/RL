"""
Full-trace demo: PPO 결정 + BM25 변화 + LLM signal + 최종 답
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import PPO

from envs.doc_reader_env import DocReaderEnv, N_DOCS, ANSWER_ACTION


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="logs/ppo_doc_reader.zip")
    ap.add_argument("--data", default="data/hotpot_doc_reader.json")
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-steps", type=int, default=4)
    ap.add_argument("--vllm-url", default="http://localhost:8002/v1")
    ap.add_argument("--vllm-model", default="Qwen/Qwen2.5-7B-Instruct")
    args = ap.parse_args()

    with open(args.data) as f:
        episodes = json.load(f)
    ep = episodes[args.idx]

    env = DocReaderEnv(
        args.data,
        max_steps=args.max_steps,
        vllm_url=args.vllm_url,
        vllm_model=args.vllm_model,
        dry_run=args.dry_run,
    )
    model = PPO.load(args.model, env=env)

    print(f"\n{'='*78}")
    print(f"Question : {ep['question']}")
    print(f"Answer   : {ep['answer']}")
    print(f"Type     : {ep['type']}")
    print(f"Gold idx : {ep['gold_indices']}")
    print(f"\nAll 10 docs (★ = gold):")
    for i, d in enumerate(ep["docs"]):
        mark = "★" if i in ep["gold_indices"] else " "
        print(f"  [{i}] {mark} {d['title']}")
    print(f"{'='*78}\n")

    env._init_episode(ep)
    obs = env._obs()
    done = False
    step = 0
    total_r = 0.0
    info = {}

    while not done:
        bm25 = obs[N_DOCS : 2 * N_DOCS]
        suff = obs[2 * N_DOCS]
        unread_top3 = sorted(
            [(i, float(bm25[i])) for i in range(N_DOCS) if env.read_mask[i] == 0],
            key=lambda x: -x[1],
        )[:3]
        step += 1

        a, _ = model.predict(obs, deterministic=True)
        a = int(a)
        print(f"[Step {step}]")
        print(f"  read_mask     : {env.read_mask}")
        print(f"  BM25 top-3    : {[(f'doc{i}', round(s,2)) for i, s in unread_top3]}")
        print(f"  llm_sufficient: {int(suff)}")
        if a == ANSWER_ACTION:
            print(f"  PPO action    : ANSWER (10)")
            obs, r, term, trunc, info = env.step(a)
            total_r += r
            print(f"  → LLM answer  : '{info.get('answer','')}'")
            print(f"  → gold answer : '{ep['answer']}'")
            print(f"  → correct     : {info.get('correct')}")
            print(f"  → reward      : {r:+.2f}")
            done = True
        else:
            mark = "★" if a in ep["gold_indices"] else " "
            print(f"  PPO action    : read doc#{a} {mark} [{ep['docs'][a]['title']}]")
            obs, r, term, trunc, info = env.step(a)
            total_r += r
            print(f"  → reward      : {r:+.2f}")
            print(f"  → new llm_suff: {info.get('llm_sufficient', '?')}")
            if term or trunc:
                done = True
        print()

    print(f"{'='*78}")
    print(f"Done.  total_return={total_r:+.2f}  success={info.get('success', False)}")


if __name__ == "__main__":
    main()
