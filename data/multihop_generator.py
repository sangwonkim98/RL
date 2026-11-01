import json
import random
import argparse
from pathlib import Path

CHAINS = [
    {"entities": ["Inception", "Christopher Nolan", "Emma Thomas", "Dunkirk"],
     "relations": ["directed_by", "spouse", "produced"]},
    {"entities": ["The Godfather", "Francis Coppola", "Eleanor Coppola", "Hearts of Darkness"],
     "relations": ["directed_by", "spouse", "directed"]},
    {"entities": ["Avatar", "James Cameron", "Suzy Amis", "The Usual Suspects"],
     "relations": ["directed_by", "spouse", "starred_in"]},
    {"entities": ["Jurassic Park", "Steven Spielberg", "Kate Capshaw", "The Color Purple"],
     "relations": ["directed_by", "spouse", "starred_in"]},
    {"entities": ["Pulp Fiction", "Quentin Tarantino", "Uma Thurman", "Kill Bill"],
     "relations": ["directed_by", "collaborated_with", "starred_in"]},
    {"entities": ["La La Land", "Damien Chazelle", "Olivia Hamilton", "First Man"],
     "relations": ["directed_by", "spouse", "produced"]},
]


def make_episode(ep_id, chain):
    e = chain["entities"]
    r = chain["relations"]
    hops = [
        {"slot": 0, "query_entity": e[0], "relation": r[0],
         "answer_entity": e[1], "gold_doc_id": f"d_{ep_id}_0"},
        {"slot": 1, "query_entity": e[1], "relation": r[1],
         "answer_entity": e[2], "gold_doc_id": f"d_{ep_id}_1"},
        {"slot": 2, "query_entity": e[2], "relation": r[2],
         "answer_entity": e[3], "gold_doc_id": f"d_{ep_id}_2"},
    ]
    docs = [
        {"id": f"d_{ep_id}_0", "text": f"{e[0]} is {r[0]} {e[1]}.",
         "entities": [e[0], e[1]]},
        {"id": f"d_{ep_id}_1", "text": f"{e[1]} {r[1]} {e[2]}.",
         "entities": [e[1], e[2]]},
        {"id": f"d_{ep_id}_2", "text": f"{e[2]} {r[2]} {e[3]}.",
         "entities": [e[2], e[3]]},
    ]
    return {
        "id": f"ep_{ep_id}",
        "question": f"What did the {r[1]} of the {r[0]} of {e[0]} {r[2]}?",
        "answer": e[3],
        "hops": hops,
        "documents": docs,
        "gold_trajectory": [0, 1, 2, 3],
    }


def generate(n, out_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    random.seed(0)
    eps = [make_episode(i, random.choice(CHAINS)) for i in range(n)]
    with open(out_path, "w") as f:
        json.dump(eps, f, indent=2)
    print(f"Generated {n} episodes -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="data/level1_fixed.json")
    args = ap.parse_args()
    generate(args.n, args.out)
