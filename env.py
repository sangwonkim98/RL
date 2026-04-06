import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
import random
from sklearn.feature_extraction.text import TfidfVectorizer

class DocNavEnv(gym.Env):
    """
    Sequential Document Navigation Environment for Multi-hop QA.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, data_path, max_docs=5, max_steps=10, feature_dim=64):
        super(DocNavEnv, self).__init__()
        
        with open(data_path, "r") as f:
            self.dataset = json.load(f)
            
        self.max_docs = max_docs
        self.max_steps = max_steps
        self.feature_dim = feature_dim
        
        # TF-IDF Vectorizer for simple features
        self.vectorizer = TfidfVectorizer(max_features=self.feature_dim)
        self._fit_vectorizer()
        
        # Update feature_dim to actual number of features found
        self.actual_feature_dim = len(self.vectorizer.get_feature_names_out())
        
        # Action space: 0 to max_docs-1 for docs, max_docs for STOP
        self.action_space = spaces.Discrete(self.max_docs + 1)
        
        # Observation space: 
        # Question embedding (actual_feature_dim)
        # All doc embeddings (max_docs * actual_feature_dim)
        # Visited mask (max_docs)
        # Current step (1)
        obs_size = self.actual_feature_dim + (self.max_docs * self.actual_feature_dim) + self.max_docs + 1
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(obs_size,), dtype=np.float32)
        
        self.current_idx = 0
        self.reset()

    def _fit_vectorizer(self):
        all_text = []
        for d in self.dataset:
            all_text.append(d["question"])
            for doc in d["documents"]:
                all_text.append(doc["text"])
        self.vectorizer.fit(all_text)

    def _get_obs(self):
        # Flattened observation
        obs = np.concatenate([
            self.q_feat,
            self.doc_feats.flatten(),
            self.visited.astype(np.float32),
            np.array([self.steps], dtype=np.float32)
        ])
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Sample an episode
        self.episode_data = random.choice(self.dataset)
        self.question = self.episode_data["question"]
        self.docs = self.episode_data["documents"][:self.max_docs]
        
        # Pad documents if less than max_docs
        while len(self.docs) < self.max_docs:
            self.docs.append({"text": "", "is_support": 0})
            
        # Precompute features
        self.q_feat = self.vectorizer.transform([self.question]).toarray().flatten()
        self.doc_feats = self.vectorizer.transform([d["text"] for d in self.docs]).toarray()
        
        self.visited = np.zeros(self.max_docs, dtype=bool)
        self.steps = 0
        self.support_found = 0
        self.total_support = sum(1 for d in self.docs if d.get("is_support", 0) == 1)
        
        return self._get_obs(), {}

    def step(self, action):
        reward = -0.05  # Step penalty
        terminated = False
        truncated = False
        
        # Action: STOP
        if action == self.max_docs:
            if self.support_found == self.total_support:
                reward = 5.0
            else:
                reward = -3.0
            terminated = True
            return self._get_obs(), reward, terminated, truncated, {}

        # Action: Select document
        doc_idx = action
        doc = self.docs[doc_idx]
        
        if self.visited[doc_idx]:
            reward -= 0.5  # Repeated selection
        else:
            self.visited[doc_idx] = True
            if doc.get("is_support", 0) == 1:
                reward += 1.0
                self.support_found += 1
                if self.support_found == self.total_support:
                    reward += 3.0
            else:
                reward -= 0.2  # Irrelevant doc
                
        self.steps += 1
        if self.steps >= self.max_steps:
            truncated = True
            
        return self._get_obs(), reward, terminated, truncated, {"support_found": self.support_found, "total": self.total_support}

    def render(self):
        pass
