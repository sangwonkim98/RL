import json
import numpy as np
from stable_baselines3 import PPO
from ppo_navigation.env import DocNavEnv
from ppo_navigation.baselines import RandomPolicy, GreedySimilarityPolicy

def run_evaluation(env, policy, num_episodes=50):
    total_rewards = []
    successes = [] # Defined as finding all support docs and stopping correctly
    
    for _ in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        support_found = 0
        total_support = 0
        
        while not (done or truncated):
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            support_found = info.get("support_found", 0)
            total_support = info.get("total", 0)
            
        total_rewards.append(episode_reward)
        if support_found == total_support and done:
            successes.append(1.0)
        else:
            successes.append(0.0)
            
    return np.mean(total_rewards), np.mean(successes)

def main():
    data_path = "ppo_navigation/data.json"
    env = DocNavEnv(data_path=data_path, max_docs=5, max_steps=10, feature_dim=64)
    
    # Load PPO
    ppo_model = PPO.load("ppo_navigation/ppo_doc_nav_model.zip", env=env)
    
    # Instantiate baselines
    random_policy = RandomPolicy(env.action_space)
    greedy_policy = GreedySimilarityPolicy(max_docs=env.max_docs, actual_feature_dim=env.actual_feature_dim)
    
    # Evaluate
    policies = {
        "PPO": ppo_model,
        "Random": random_policy,
        "Greedy Similarity": greedy_policy
    }
    
    results = {}
    for name, policy in policies.items():
        avg_reward, avg_success = run_evaluation(env, policy)
        results[name] = {"reward": avg_reward, "success_rate": avg_success}
        print(f"{name}: Avg Reward: {avg_reward:.2f}, Success Rate: {avg_success:.2%}")
        
    with open("ppo_navigation/metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Metrics saved to ppo_navigation/metrics.json")

if __name__ == "__main__":
    main()
