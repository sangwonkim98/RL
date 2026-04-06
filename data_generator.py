import json
import random

def generate_synthetic_data(num_samples=20, num_docs=5):
    data = []
    for i in range(num_samples):
        # Generate some topics for questions
        topics = ["AI", "Paris", "Coffee", "Quantum Mechanics", "Dogs", "Pizza", "History", "Math", "Biology", "Space"]
        topic = random.choice(topics)
        
        # In a real case, question/docs would be more related
        question = f"What is related to {topic}?"
        
        documents = []
        num_support = random.randint(1, 2)
        
        # Add support documents
        for j in range(num_support):
            documents.append({
                "id": f"doc_{i}_{j}",
                "title": f"Support Doc for {topic} {j}",
                "text": f"This document contains important info about {topic} and its properties.",
                "is_support": 1
            })
            
        # Add irrelevant documents
        for j in range(num_docs - num_support):
            other_topic = random.choice([t for t in topics if t != topic])
            documents.append({
                "id": f"doc_{i}_{j+num_support}",
                "title": f"Irrelevant Doc {j}",
                "text": f"This document is about something else like {other_topic}.",
                "is_support": 0
            })
            
        random.shuffle(documents)
        
        data.append({
            "id": f"ex_{i}",
            "question": question,
            "answer": f"Answer about {topic}",
            "documents": documents
        })
        
    with open("ppo_navigation/data.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"Generated {num_samples} samples in ppo_navigation/data.json")

if __name__ == "__main__":
    generate_synthetic_data()
