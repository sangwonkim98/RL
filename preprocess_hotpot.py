import json
import os
import argparse
import random

def preprocess_hotpot_style(input_path, output_path, max_distractors=3):
    """
    Converts HotpotQA-style JSON to the DocNavEnv format.
    
    HotpotQA typical structure:
    [
      {
        "_id": "...",
        "question": "...",
        "answer": "...",
        "supporting_facts": [["Title A", sentence_idx], ...],
        "context": [["Title A", ["sent0", "sent1", ...]], ["Title B", [...]], ...]
      }
    ]
    """
    if not os.path.exists(input_path):
        print(f"Source file {input_path} not found. Skipping real preprocessing.")
        return

    with open(input_path, "r") as f:
        raw_data = json.load(f)

    processed_data = []

    for item in raw_data:
        # 1. Extract basic info
        # TODO: Adjust keys if your specific version of HotpotQA uses different names
        question = item.get("question", "")
        answer = item.get("answer", "")
        item_id = item.get("_id", item.get("id", "unknown"))
        
        # 2. Identify supporting titles
        # supporting_facts is a list of [title, sentence_index]
        support_titles = set([fact[0] for fact in item.get("supporting_facts", [])])
        
        # 3. Build document list from context
        documents = []
        context = item.get("context", [])
        
        for title, sentences in context:
            text = " ".join(sentences)
            is_support = 1 if title in support_titles else 0
            
            documents.append({
                "title": title,
                "text": text,
                "is_support": is_support
            })

        # 4. Optional: Filter/Limit distractors to keep document pool manageable
        # (The environment uses max_docs, so we should ensure we have enough or cap them)
        support_docs = [d for d in documents if d["is_support"] == 1]
        distractor_docs = [d for d in documents if d["is_support"] == 0]
        
        # Shuffle and take a subset of distractors if needed
        random.shuffle(distractor_docs)
        selected_docs = support_docs + distractor_docs[:max_distractors]
        random.shuffle(selected_docs)

        processed_data.append({
            "id": item_id,
            "question": question,
            "answer": answer,
            "documents": selected_docs
        })

    with open(output_path, "w") as f:
        json.dump(processed_data, f, indent=2)
    
    print(f"Successfully processed {len(processed_data)} items to {output_path}")

def create_dummy_hotpot_file(path):
    """Creates a tiny sample file to demonstrate the script works."""
    dummy = [
      {
        "_id": "hp_001",
        "question": "Who directed the movie involving a detective and a robot?",
        "answer": "John Doe",
        "supporting_facts": [["Movie Alpha", 0], ["Director Bio", 0]],
        "context": [
          ["Movie Alpha", ["Movie Alpha is a 2020 film about a detective and a robot."]],
          ["Director Bio", ["John Doe is a famous director who made Movie Alpha."]],
          ["Random News", ["The weather today is sunny with a chance of rain."]],
          ["Cooking Tips", ["To make a great omelette, use fresh eggs and butter."]]
        ]
      }
    ]
    with open(path, "w") as f:
        json.dump(dummy, f, indent=2)
    print(f"Created dummy HotpotQA file at {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess HotpotQA-style data for DocNavEnv.")
    parser.get_default("input_path")
    parser.add_argument("--input", type=str, default="hotpot_train_v1.1.json", help="Path to raw HotpotQA JSON")
    parser.add_argument("--output", type=str, default="ppo_navigation/hotpot_processed.json", help="Path to save processed JSON")
    parser.add_argument("--dummy", action="store_true", help="Create a dummy input file for testing")
    
    args = parser.parse_args()

    if args.dummy:
        create_dummy_hotpot_file(args.input)

    preprocess_hotpot_style(args.input, args.output)
