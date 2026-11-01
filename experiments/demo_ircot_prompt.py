"""
Demo: PPO Router → IRCoT-style prompt → LLM (vLLM :8002) or dry-run

Usage:
  python -m experiments.demo_ircot_prompt --idx 0 --dry-run
  python -m experiments.demo_ircot_prompt --idx 0
"""
import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from envs.path_router_env import PathRouterEnv, STOP, RETRIEVE_SEED, RETRIEVE_BRIDGE

ACTION_NAMES = {0: "retrieve_seed", 1: "retrieve_bridge", 2: "retrieve_relation", 3: "stop"}

ACTION_INSTRUCTION = {
    0: "Identify the first key entity in the question and write a search query to find the seed document.",
    1: "Using the seed entity found above, write a search query to find the bridging document.",
    2: "Retrieve a relation or attribute of the previously discovered entity.",
    3: "You now have sufficient evidence. Synthesize the evidence and state the final answer.",
}

PROMPT_TEMPLATE = """\
You are a multi-hop reasoning assistant. Follow the routing instruction exactly.

Question: {question}

Collected evidence:
{evidence}

[Routing instruction] {action_name}: {action_instruction}

Generate the following:
Reasoning: <one or two sentences explaining what you are looking for>
Search query: <a concise query (≤10 words) to retrieve the next document>
"""

STOP_TEMPLATE = """\
You are a multi-hop reasoning assistant.

Question: {question}

Collected evidence:
{evidence}

[Routing instruction] stop: All evidence is collected. Provide the final answer.

Answer: <your answer>
"""


def build_prompt(question, evidence_docs, action):
    evidence_str = "\n".join(
        f"[Doc {i+1}] ({d['title']}) {d['text'][:300]}..."
        for i, d in enumerate(evidence_docs)
    ) or "(no evidence yet)"

    if action == STOP:
        return STOP_TEMPLATE.format(question=question, evidence=evidence_str)
    return PROMPT_TEMPLATE.format(
        question=question,
        evidence=evidence_str,
        action_name=ACTION_NAMES[action],
        action_instruction=ACTION_INSTRUCTION[action],
    )


async def call_vllm(prompt, base_url, model_name):
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key="dummy", base_url=base_url)
    resp = await client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


async def run_demo(args):
    with open(args.data) as f:
        episodes = json.load(f)

    env = PathRouterEnv(args.data)
    model = PPO.load(args.model, env=env)

    ep = episodes[args.idx]
    q = ep["question"]
    print(f"\n{'='*72}")
    print(f"Question : {q}")
    print(f"Answer   : {ep['answer']}")
    print(f"Type     : {ep['type']}")
    print(f"Gold docs: {ep['gold_titles']}")
    print(f"Gold path: {ep['gold_actions']}")
    print(f"{'='*72}\n")

    # 에피소드 초기화
    env.ep = ep
    env._init_episode(ep)
    obs = env._obs()

    collected_docs = []
    step = 0
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        print(f"[Step {step+1}] PPO Router → {ACTION_NAMES[action]}")

        # 골드 doc 시뮬레이션 검색 (retrieve 액션일 때)
        if action in (RETRIEVE_SEED, RETRIEVE_BRIDGE):
            hop_map = {RETRIEVE_SEED: 0, RETRIEVE_BRIDGE: 1}
            hop_idx = hop_map[action]
            if hop_idx < len(ep["gold_titles"]):
                title = ep["gold_titles"][hop_idx]
                doc = next((d for d in ep["documents"] if d["title"] == title), None)
                if doc:
                    collected_docs.append(doc)
                    print(f"         Retrieved: [{title}]")

        prompt = build_prompt(q, collected_docs, action)
        print(f"\n--- Prompt ---\n{prompt}")

        if not args.dry_run:
            try:
                out = await asyncio.wait_for(
                    call_vllm(prompt, args.vllm_url, args.vllm_model),
                    timeout=60,
                )
                print(f"--- LLM Output ---\n{out}\n")
            except Exception as e:
                print(f"[LLM error: {e}]\n")
        else:
            print("[dry-run: LLM call skipped]\n")

        obs, r, term, trunc, info = env.step(action)
        step += 1
        if term or trunc:
            done = True

    success = info.get("success", False)
    print(f"{'='*72}")
    print(f"Done. Success={success}  hop_status={info['hop_status']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="logs/ppo_router.zip")
    ap.add_argument("--data", default="data/hotpot_path_data.json")
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="skip LLM call, print prompt only")
    ap.add_argument("--vllm-url", default="http://localhost:8002/v1")
    ap.add_argument("--vllm-model", default="Qwen/Qwen2.5-7B-Instruct")
    args = ap.parse_args()
    asyncio.run(run_demo(args))


if __name__ == "__main__":
    main()
