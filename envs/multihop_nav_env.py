import json
import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces

RETRIEVE_ORIGINAL, RETRIEVE_BRIDGE, RETRIEVE_RELATION, STOP = 0, 1, 2, 3
N_HOPS = 3
N_ACTIONS = 4


class MultihopNavEnv(gym.Env):
    """
    4-action MDP for multi-hop evidence navigation (Level 1: fixed order).

    obs (9-dim): hop_status(3) + last_action_onehot(4) + step_norm(1) + last_retrieval_ok(1)
    """
    metadata = {"render_modes": []}

    def __init__(self, data_path, max_steps=8):
        super().__init__()
        with open(data_path) as f:
            self.episodes = json.load(f)
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(9,), dtype=np.float32
        )
        self.ep = None
        self.hop_status = None
        self.last_action = None
        self.last_retrieval_ok = 0
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.ep = random.choice(self.episodes)
        self.hop_status = [0, 0, 0]
        self.last_action = None
        self.last_retrieval_ok = 0
        self.step_count = 0
        return self._obs(), {}

    def step(self, action):
        action = int(action)
        reward = -0.1  # step cost
        terminated = False
        truncated = False
        retrieval_ok = 0

        if action == STOP:
            if all(s == 1 for s in self.hop_status):
                reward += 3.0
            else:
                reward -= 2.0
            terminated = True
        else:
            # Level 1: action k fills hop k iff hop k not filled AND previous hops filled
            k = action
            if self.hop_status[k] == 1:
                reward -= 0.5  # already filled = failed retrieval
            elif k > 0 and self.hop_status[k - 1] != 1:
                reward -= 0.5  # out-of-order = failed retrieval
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
            "success": terminated and all(s == 1 for s in self.hop_status),
            "step": self.step_count,
        }
        return self._obs(), reward, terminated, truncated, info

    def _obs(self):
        obs = np.zeros(9, dtype=np.float32)
        obs[0:3] = self.hop_status
        if self.last_action is not None:
            obs[3 + self.last_action] = 1.0
        obs[7] = self.step_count / self.max_steps
        obs[8] = float(self.last_retrieval_ok)
        return obs
