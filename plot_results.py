import json
import matplotlib.pyplot as plt
import numpy as np

def plot():
    try:
        with open("ppo_navigation/metrics.json", "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        print("Metrics file not found. Run evaluation first.")
        return
        
    names = list(results.keys())
    rewards = [results[n]["reward"] for n in names]
    success_rates = [results[n]["success_rate"] for n in names]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Avg Reward Plot
    ax1.bar(names, rewards, color=['blue', 'gray', 'green'])
    ax1.set_title("Average Episode Reward")
    ax1.set_ylabel("Reward")
    
    # Success Rate Plot
    ax2.bar(names, success_rates, color=['blue', 'gray', 'green'])
    ax2.set_title("Success Rate (All Support Found & STOP)")
    ax2.set_ylabel("Success Rate")
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig("ppo_navigation/evaluation_results.png")
    print("Results plot saved to ppo_navigation/evaluation_results.png")

if __name__ == "__main__":
    plot()
