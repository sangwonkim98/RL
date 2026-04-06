import os
from stable_baselines3 import PPO
from ppo_navigation.env import DocNavEnv

def train():
    data_path = "ppo_navigation/data.json"
    env = DocNavEnv(data_path=data_path, max_docs=5, max_steps=10, feature_dim=64)
    
    # Simple MLP policy
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_navigation/logs/")
    
    print("Starting training...")
    model.learn(total_timesteps=5000)
    
    model_path = "ppo_navigation/ppo_doc_nav_model.zip"
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Quick eval
    obs, info = env.reset()
    done = False
    total_reward = 0
    print("\nStarting sample evaluation episode:")
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        print(f"Action: {action}, Reward: {reward:.2f}")
    print(f"Total reward: {total_reward:.2f}")

if __name__ == "__main__":
    train()
