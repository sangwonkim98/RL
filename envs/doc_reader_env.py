"""
DocReaderEnv: 11-action MDP for LLM-in-loop document reader

Action (11):
  0~9 : read doc i
  10  : answer (LLM이 최종 답 생성)

Observation (22-dim):
  [0:10]   read_mask           (어떤 doc 읽었는지)
  [10:20]  bm25_scores         (현재 evidence 누적 기준 BM25, 매 step 재계산)
  [20]     llm_sufficient      (LLM이 "지금 답 가능?" 0/1)
  [21]     step_norm

Reward:
  - read gold doc      : +1.0
  - read distractor    : -0.3
  - 이미 읽은 doc 재읽기 : -0.5
  - answer correct     : +5.0
  - answer wrong       : -3.0
  - step cost (매 step): -0.1
"""
import json
import math
import random
import re
from collections import Counter

import gymnasium as gym
import numpy as np
from gymnasium import spaces

N_DOCS = 10
N_ACTIONS = N_DOCS + 1
ANSWER_ACTION = N_DOCS
# obs: read_mask(10) + bm25(10) + bm25_unread_argmax_onehot(10) + llm_sufficient(1) + step_norm(1)
OBS_DIM = N_DOCS * 3 + 2

_WORD_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return _WORD_RE.findall(text.lower())


def normalize_answer(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def answer_correct(pred, gold):
    p, g = normalize_answer(pred), normalize_answer(gold)
    if not g:
        return False
    if g in p:
        return True
    pw, gw = set(p.split()), set(g.split())
    if gw and len(pw & gw) / len(gw) >= 0.5:
        return True
    return False


class BM25Lite:
    """Minimal BM25 (k1=1.5, b=0.75) — corpus 작아서 단순 구현 충분"""

    def __init__(self, docs_tokens, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.N = len(docs_tokens)
        self.docs = [Counter(t) for t in docs_tokens]
        self.dl = [sum(c.values()) for c in self.docs]
        self.avgdl = sum(self.dl) / max(self.N, 1) if self.N else 0
        df = Counter()
        for c in self.docs:
            for w in c:
                df[w] += 1
        self.idf = {
            w: math.log((self.N - n + 0.5) / (n + 0.5) + 1.0)
            for w, n in df.items()
        }

    def scores(self, q_tokens):
        out = np.zeros(self.N, dtype=np.float32)
        if not q_tokens:
            return out
        for i, doc in enumerate(self.docs):
            s = 0.0
            for w in q_tokens:
                if w not in doc:
                    continue
                f = doc[w]
                idf = self.idf.get(w, 0.0)
                denom = f + self.k1 * (
                    1 - self.b + self.b * (self.dl[i] / (self.avgdl + 1e-9))
                )
                s += idf * (f * (self.k1 + 1)) / (denom + 1e-9)
            out[i] = s
        return out


class DocReaderEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        data_path,
        max_steps=6,
        vllm_url="http://localhost:8002/v1",
        vllm_model="Qwen/Qwen2.5-7B-Instruct",
        llm_cache=None,
        dry_run=False,
        text_clip=400,
    ):
        super().__init__()
        with open(data_path) as f:
            self.episodes = json.load(f)
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.vllm_url = vllm_url
        self.vllm_model = vllm_model
        self.dry_run = dry_run
        self.text_clip = text_clip
        self._client = None  # lazy init (pickling safe)
        # 캐시 외부 주입 가능 — vec env에서 공유하면 좋음
        self.llm_cache = llm_cache if llm_cache is not None else {}

        # 에피소드 상태
        self.ep = None
        self.read_mask = None
        self.collected_text = ""
        self.bm25_scores = None
        self.llm_sufficient = 0
        self.step_count = 0

    # ── LLM 호출 ─────────────────────────────────────────────────────────────
    def _get_client(self):
        if self._client is None and not self.dry_run:
            from openai import OpenAI
            self._client = OpenAI(api_key="dummy", base_url=self.vllm_url)
        return self._client

    def _build_evidence_str(self):
        parts = []
        for i in range(N_DOCS):
            if self.read_mask[i]:
                d = self.ep["docs"][i]
                parts.append(f"[Doc {i}] {d['title']}: {d['text'][:self.text_clip]}")
        return "\n".join(parts)

    def _llm_sufficient_signal(self):
        """현재 evidence가 답하기 충분한지 LLM에 묻기 (0/1)"""
        if sum(self.read_mask) == 0:
            return 0

        key = ("suff", self.ep["id"], tuple(self.read_mask))
        if key in self.llm_cache:
            return self.llm_cache[key]

        if self.dry_run:
            # 휴리스틱: 두 gold 다 읽으면 1
            n_gold = sum(self.read_mask[i] for i in self.ep["gold_indices"])
            result = 1 if n_gold >= 2 else 0
            self.llm_cache[key] = result
            return result

        evidence = self._build_evidence_str()
        prompt = (
            f"Question: {self.ep['question']}\n\n"
            f"Evidence:\n{evidence}\n\n"
            "Is the evidence above sufficient to answer the question? "
            "Respond with exactly one digit and nothing else: 0 (insufficient) or 1 (sufficient)."
        )
        try:
            resp = self._get_client().chat.completions.create(
                model=self.vllm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2,
                temperature=0.0,
            )
            txt = (resp.choices[0].message.content or "").strip()
            result = 1 if txt.startswith("1") else 0
        except Exception:
            result = 0
        self.llm_cache[key] = result
        return result

    def _llm_answer(self):
        """최종 답 생성"""
        if sum(self.read_mask) == 0:
            return ""

        key = ("ans", self.ep["id"], tuple(self.read_mask))
        if key in self.llm_cache:
            return self.llm_cache[key]

        if self.dry_run:
            n_gold = sum(self.read_mask[i] for i in self.ep["gold_indices"])
            result = self.ep["answer"] if n_gold >= 2 else "unknown"
            self.llm_cache[key] = result
            return result

        evidence = self._build_evidence_str()
        prompt = (
            f"Question: {self.ep['question']}\n\n"
            f"Evidence:\n{evidence}\n\n"
            "Provide ONLY the short final answer (no explanation, no sentence).\n"
            "Answer:"
        )
        try:
            resp = self._get_client().chat.completions.create(
                model=self.vllm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=32,
                temperature=0.0,
            )
            result = (resp.choices[0].message.content or "").strip()
        except Exception:
            result = ""
        self.llm_cache[key] = result
        return result

    # ── BM25 업데이트 ────────────────────────────────────────────────────────
    def _update_bm25(self):
        docs_tok = [
            tokenize(d["title"] + " " + d["text"][: self.text_clip])
            for d in self.ep["docs"]
        ]
        bm25 = BM25Lite(docs_tok)
        q_tok = tokenize(self.ep["question"] + " " + self.collected_text)
        s = bm25.scores(q_tok)
        # 0~1 정규화
        lo, hi = float(s.min()), float(s.max())
        if hi - lo < 1e-6:
            self.bm25_scores = np.zeros(N_DOCS, dtype=np.float32)
        else:
            self.bm25_scores = ((s - lo) / (hi - lo)).astype(np.float32)

    # ── Gym API ─────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.ep = random.choice(self.episodes)
        self._init_episode(self.ep)
        return self._obs(), {}

    def _init_episode(self, ep):
        self.ep = ep
        self.read_mask = [0] * N_DOCS
        self.collected_text = ""
        self.llm_sufficient = 0
        self.step_count = 0
        self._update_bm25()

    def step(self, action):
        action = int(action)
        reward = -0.2  # step cost (높임 → 빨리 끝내게)
        terminated = False
        truncated = False
        info = {}

        if action == ANSWER_ACTION:
            pred = self._llm_answer()
            correct = answer_correct(pred, self.ep["answer"])
            reward += 8.0 if correct else -4.0
            terminated = True
            info["answer"] = pred
            info["correct"] = correct
            info["success"] = correct
        else:
            if self.read_mask[action] == 1:
                reward -= 0.7  # 재읽기 페널티 강화
            else:
                self.read_mask[action] = 1
                if action in self.ep["gold_indices"]:
                    reward += 1.5  # gold 보상 강화
                else:
                    reward -= 0.4
                # evidence 누적 + BM25 / sufficiency 신호 갱신
                self.collected_text += " " + self.ep["docs"][action]["text"][: self.text_clip]
                self._update_bm25()
                self.llm_sufficient = self._llm_sufficient_signal()

        self.step_count += 1
        if not terminated and self.step_count >= self.max_steps:
            truncated = True

        info["read_mask"] = list(self.read_mask)
        info["step"] = self.step_count
        info["llm_sufficient"] = int(self.llm_sufficient)
        if "success" not in info:
            info["success"] = False
        return self._obs(), reward, terminated, truncated, info

    def _obs(self):
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[0:N_DOCS] = self.read_mask
        obs[N_DOCS:2 * N_DOCS] = self.bm25_scores
        # 미독 docs 중 BM25 최고 위치 one-hot
        unread_scores = np.where(
            np.array(self.read_mask) == 0, self.bm25_scores, -1.0
        )
        if unread_scores.max() >= 0:
            top_idx = int(np.argmax(unread_scores))
            obs[2 * N_DOCS + top_idx] = 1.0
        obs[3 * N_DOCS] = float(self.llm_sufficient)
        obs[3 * N_DOCS + 1] = self.step_count / self.max_steps
        return obs
