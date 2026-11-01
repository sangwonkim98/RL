"""
PathRouterEnv: gold-action-supervised 4-action MDP
obs (9-dim): hop_status(3) + last_action_onehot(4) + step_norm(1) + last_retrieval_ok(1)
gold_actions per episode: [0,1,3] (retrieve_seed, retrieve_bridge, stop)
"""
import json
import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces

RETRIEVE_SEED, RETRIEVE_BRIDGE, RETRIEVE_RELATION, STOP = 0, 1, 2, 3
N_ACTIONS = 4
OBS_DIM = 9
HOP_IDX = {RETRIEVE_SEED: 0, RETRIEVE_BRIDGE: 1, RETRIEVE_RELATION: 2}


class PathRouterEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, data_path, max_steps=8):
        super().__init__()
        with open(data_path) as f:
            self.episodes = json.load(f)
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.ep = None
        self.gold_actions = None
        self.n_required = None
        self.hop_status = None
        self.last_action = None
        self.last_retrieval_ok = 0
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.ep = random.choice(self.episodes)
        self._init_episode(self.ep)
        return self._obs(), {}

    def _init_episode(self, ep):
        self.gold_actions = ep["gold_actions"]
        self.n_required = sum(1 for a in self.gold_actions if a != STOP)
        self.hop_status = [0, 0, 0]
        self.last_action = None
        self.last_retrieval_ok = 0
        self.step_count = 0

    def step(self, action):
        action = int(action)
        reward = -0.1
        terminated = False
        truncated = False
        retrieval_ok = 0

        gold_now = self.gold_actions[min(self.step_count, len(self.gold_actions) - 1)]
        n_filled = sum(self.hop_status)

        if action == STOP:
            if n_filled >= self.n_required:
                reward += 3.0
            else:
                reward -= 2.0
            terminated = True
        else:
            k = HOP_IDX[action]
            if self.hop_status[k] == 1:
                reward -= 0.5  # 이미 채워진 hop
            elif action != gold_now:
                reward -= 0.5  # gold 순서 위반
            else:
                self.hop_status[k] = 1
                reward += 1.0
                retrieval_ok = 1

        self.last_action = action
        self.last_retrieval_ok = retrieval_ok
        self.step_count += 1
        if not terminated and self.step_count >= self.max_steps:
            truncated = True

        info = {
            "hop_status": list(self.hop_status),
            "gold_action": gold_now,
            "success": terminated and n_filled >= self.n_required,
            "step": self.step_count,
        }
        return self._obs(), reward, terminated, truncated, info

    def _obs(self):
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[0:3] = self.hop_status
        if self.last_action is not None:
            obs[3 + self.last_action] = 1.0
        obs[7] = self.step_count / self.max_steps
        obs[8] = float(self.last_retrieval_ok)
        return obs
