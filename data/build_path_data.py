"""
HotpotQA distractor → gold action path data
Bridge/Comparison 모두 2-hop: [retrieve_seed(0), retrieve_bridge(1), stop(3)]
"""
import json
import argparse
from pathlib import Path

RETRIEVE_SEED, RETRIEVE_BRIDGE, RETRIEVE_RELATION, STOP = 0, 1, 2, 3


def build_episode(sample):
    # 순서 있는 gold titles (supporting_facts 순서 유지)
    seen = set()
    gold_titles = []
    for title, _ in sample["supporting_facts"]:
        if title not in seen:
            seen.add(title)
            gold_titles.append(title)
        if len(gold_titles) == 2:
            break

    docs = [
        {"title": title, "text": " ".join(sents)}
        for title, sents in sample["context"]
    ]

    return {
        "id": sample["_id"],
        "question": sample["question"],
        "answer": sample["answer"],
        "type": sample["type"],
        "gold_titles": gold_titles,
        "gold_actions": [RETRIEVE_SEED, RETRIEVE_BRIDGE, STOP],
        "documents": docs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/hotpot_dev_distractor.json")
    ap.add_argument("--out", default="data/hotpot_path_data.json")
    ap.add_argument("--n", type=int, default=None)
    args = ap.parse_args()

    with open(args.src) as f:
        raw = json.load(f)

    if args.n:
        raw = raw[: args.n]

    episodes = [build_episode(s) for s in raw]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(episodes, f)

    types = {}
    for e in episodes:
        types[e["type"]] = types.get(e["type"], 0) + 1
    print(f"Built {len(episodes)} episodes → {args.out}")
    print(f"Type distribution: {types}")


if __name__ == "__main__":
    main()
