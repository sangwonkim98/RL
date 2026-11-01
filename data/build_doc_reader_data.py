"""
HotpotQA distractor → doc reader format
각 episode: 10 docs (gold 2 + distractor 8), gold_indices가 reward supervision
"""
import json
import argparse
from pathlib import Path


def build_episode(sample):
    seen = set()
    gold_titles = []
    for title, _ in sample["supporting_facts"]:
        if title not in seen:
            seen.add(title)
            gold_titles.append(title)

    docs = [
        {"title": title, "text": " ".join(sents)}
        for title, sents in sample["context"]
    ]
    gold_indices = sorted(
        i for i, d in enumerate(docs) if d["title"] in gold_titles
    )

    return {
        "id": sample["_id"],
        "question": sample["question"],
        "answer": sample["answer"],
        "type": sample["type"],
        "docs": docs,
        "gold_indices": gold_indices,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/hotpot_dev_distractor.json")
    ap.add_argument("--out", default="data/hotpot_doc_reader.json")
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    with open(args.src) as f:
        raw = json.load(f)
    raw = raw[: args.n]

    episodes = [build_episode(s) for s in raw]
    # gold 2개 정확히 발견된 것만 (가끔 title mismatch 있을 수 있음)
    episodes = [e for e in episodes if len(e["gold_indices"]) == 2]
    # 항상 10 docs인지도 확인
    episodes = [e for e in episodes if len(e["docs"]) == 10]

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
