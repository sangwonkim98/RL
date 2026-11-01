"""
DocReaderEnv 변형: bm25_argmax_unread one-hot 제거 → obs 22-dim
"PPO가 강한 휴리스틱 hint 없이도 학습 가능한가?" 검증용
"""
import json
import random

import numpy as np
from gymnasium import spaces

from envs.doc_reader_env import (
    DocReaderEnv as _Base,
    N_DOCS, N_ACTIONS, ANSWER_ACTION, BM25Lite, tokenize,
)

OBS_DIM_NO = N_DOCS * 2 + 2  # read_mask(10) + bm25(10) + llm_suff(1) + step(1)


class DocReaderEnvNoArgmax(_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM_NO,), dtype=np.float32
        )

    def _obs(self):
        obs = np.zeros(OBS_DIM_NO, dtype=np.float32)
        obs[0:N_DOCS] = self.read_mask
        obs[N_DOCS:2 * N_DOCS] = self.bm25_scores
        obs[2 * N_DOCS] = float(self.llm_sufficient)
        obs[2 * N_DOCS + 1] = self.step_count / self.max_steps
        return obs
