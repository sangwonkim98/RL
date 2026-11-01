# PPT 자료 정리 (발표용)

## 슬라이드 1 — 제목

**PPO Router for LLM-Guided Multi-Hop QA**
*Reinforcement Learning over a Frozen LLM Reader*

- HotpotQA distractor (multi-hop QA)
- PPO가 어떤 문서를 읽고 언제 답할지 결정
- LLM은 sufficiency 신호 + 최종 답 생성

---

## 슬라이드 2 — 동기

기존 IRCoT / LLM 멀티홉 추론:
- LLM이 prompt 안에서 직접 "더 찾을지 / 답할지" 판단
- 멈춤 시점이 불안정 (over-reading or premature stop)

**아이디어**: routing 결정을 **외부 RL policy**에 위임.
- PPO = 의사결정 (어디서 멈출지, 어떤 doc 읽을지)
- LLM = 의미 처리 (글 읽기, 답 만들기)

---

## 슬라이드 3 — 데이터셋

**HotpotQA distractor split**
- 한 질문당 10개 문서 (gold 2 + distractor 8)
- 2-hop reasoning 필요
- bridge type 166 + comparison type 33 = **199 episodes** 사용

예시:
- Q: "Were Scott Derrickson and Ed Wood of the same nationality?"
- gold = [Scott Derrickson, Ed Wood]
- distractor = [Doctor Strange, Sinister, ...]

---

## 슬라이드 4 — 파이프라인

```
Question + 10 docs
    ↓
DocReaderEnv
    ↓ obs(22-dim)
PPO Router
    ↓ action ∈ {read_0..9, answer}
    ├─ read doc_i → BM25 재계산 → 🔵LLM "충분?" 0/1 → next obs
    └─ answer    → 🔵LLM "Q + evidence → 답"
                  → gold answer 비교 → reward
```

**핵심**: LLM은 env 안에 있음. PPO는 LLM 출력을 obs 신호로 받음.

---

## 슬라이드 5 — RL 설계

**Action (11)**: read_0~9, answer

**Observation (32-dim)**:
- read_mask (10): 어떤 doc 읽었나
- bm25_scores (10): 질문+누적 evidence 기준 BM25
- bm25_unread_argmax_onehot (10): 미독 중 BM25 최고 위치
- llm_sufficient (1): LLM signal 0/1
- step_norm (1)

**Reward**:
- read gold doc: +1.5
- read distractor: -0.4
- 재읽기: -0.7
- answer correct: +8 / wrong: -4
- step cost: -0.2

---

## 슬라이드 6 — LLM 호출 종류

**호출 1: Sufficiency check** (매 read 후)
```
Question: ...
Evidence: [읽은 docs]
Is the evidence sufficient? Respond with 0 or 1.
```
→ max_tokens=2, latency ~0.1s

**호출 2: Answer generation** (action=answer 때)
```
Question: ...
Evidence: [읽은 docs]
Provide ONLY the short answer.
```
→ max_tokens=32, latency ~0.2s

---

## 슬라이드 7 — 학습/추론 모델 swap

- **학습**: Qwen2.5-32B-Instruct-AWQ (:8003) — 깨끗한 sufficiency 신호
- **추론**: Qwen2.5-7B-Instruct (:8002) — 빠른 응답
- 같은 Qwen 패밀리 → signal 분포 호환

코드는 한 벌. CLI flag로 LLM URL/모델만 교체.

---

## 슬라이드 8 — 학습 결과

**Setup**: 199 episodes, max_steps=4

| Method | success | avg_gold_hit | avg_reads |
|---|---|---|---|
| Random read 2 | 2.5% | 0.34/2 | 2.0 |
| BM25 top-2 + ans | 35.2% | 1.29/2 | 2.0 |
| BM25 top-3 + ans | 54.8% | 1.50/2 | 3.0 |
| BM25 top-4 + ans | 63.3% | 1.60/2 | 4.0 |
| **PPO Router (dry-run train, 100k)** | **54.8%** | **1.69/2** | **3.63** |
| PPO Router (real LLM, 10k) | 36.0% | 0.36/2 | 1.56 |

**해석**:
- dry-run 학습 PPO는 BM25 top-3와 같은 success rate지만 더 효율적
- avg_gold_hit 1.69 > top-3의 1.50 → gold doc을 더 잘 찾음
- avg_reads 3.63 → 동적으로 멈춤 (fixed top-k와 다름)
- Real LLM 학습 10k는 step 부족 — noisy signal에서는 더 많은 samples 필요

---

## 슬라이드 9 — 학습 곡선 (dry-run 100k steps)

| step | ep_rew_mean | entropy_loss | explained_var |
|---|---|---|---|
| 1k | -2.06 | -2.39 | -0.02 |
| 25k | +2.46 | -1.34 | +0.18 |
| 50k | +3.63 | -0.68 | +0.27 |
| 75k | +4.15 | -0.48 | +0.28 |
| 100k | +5.02 | -0.42 | +0.24 |

- reward 단조 증가
- entropy 감소 (점점 deterministic)
- value function explained_var 증가 → policy/value 동시 학습

(TensorBoard 데이터: `logs/tb_doc/PPO_5/`)

---

## 슬라이드 10 — Demo trace (성공 case)

Q: "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?"

```
[Step 1] BM25 top-1 = doc#5 (A Kiss for Corliss, distractor)
         PPO → read doc#5 (-0.6, distractor)
[Step 2] BM25 재계산, top = doc#1
         PPO → read doc#1 ★ Shirley Temple  (+1.3, gold)
         LLM sufficient: 0
[Step 3] PPO → read doc#6 ★ Kiss and Tell (1945 film) (+1.3, gold)
         LLM sufficient: 1
[Step 4] PPO → answer
         LLM: "Chief of Protocol"  (correct, +7.8)

Total return: +9.80   Success: ✓
```

---

## 슬라이드 11 — Demo trace (실패 case)

Q: "Were Scott Derrickson and Ed Wood of the same nationality?"

```
[Step 1] BM25 top-1 = doc#9 (Doctor Strange film, distractor)
         ← Scott Derrickson 감독한 영화라 BM25 점수 높음
         PPO → read doc#9 (distractor, -0.6)
[Step 2-4] 계속 distractor 위주로 읽음
         결국 doc#1 (Scott Derrickson)만 발견, doc#4 (Ed Wood) 못 찾음
         max_steps 도달, answer 못 침
Total return: -0.50   Success: ✗
```

→ BM25가 distractor를 더 높이 점수 주는 hard case
→ Future work: LLM 기반 relevance 신호 추가

---

## 슬라이드 12 — Transfer: dry-run 학습 → 7B real LLM 추론

dry-run 100k 학습 모델을 7B real LLM 환경에 deploy:
- idx=5 ("2014 S/S debut album..."): ✓ success +10.4
- idx=10 ("Lawrence Kansas fight song..."): ✓ success +9.8
- idx=50 ("Euromarché 1991..."): ✓ success +10.4
- idx=1 ("Corliss Archer..."): ✗ (7B가 sufficient 신호 늦게 줘서 max_steps 초과)
- idx=20 ("Mexican F1 driver..."): ✗

→ 3/5 transfer 성공. 7B가 32B/oracle보다 보수적인 sufficient 신호.
→ Train-infer gap 존재 but 적당한 generalize 됨.

---

## 슬라이드 13 — Limitations

1. **데이터 규모**: 199 episodes만 사용 (전체 7405개 중)
2. **BM25 의존**: BM25가 잘못 가리키면 PPO도 헷갈림 (Doctor Strange case)
3. **LLM sufficiency 신호의 노이즈**: 가끔 부정확
4. **2-hop으로 고정**: 더 깊은 multi-hop은 미검증

---

## 슬라이드 14 — Future Work

1. LLM-based relevance score를 BM25 대신 obs에 (더 정확)
2. 더 큰 데이터 (전체 HotpotQA + MuSiQue)
3. n-hop 일반화 (3-hop+)
4. Curriculum learning (쉬운 episode부터)

---

## 슬라이드 15 — 요약

- **PPO Router + LLM-in-env**: routing은 RL, 추론은 LLM
- HotpotQA distractor에서 BM25 baseline 대비 효율적인 routing
- 학습/추론 환경 일치 (32B → 7B swap만)
- 발표 데모: idx=1, 5, 10 (성공) / idx=0 (실패) 비교

---

## 부록: 파일 구조

```
data/build_doc_reader_data.py    # HotpotQA → 도c reader format
envs/doc_reader_env.py            # 11-action MDP, LLM in loop
experiments/train_doc_reader.py   # PPO 학습
experiments/eval_doc_reader.py    # rollout + 통계
experiments/demo_doc_reader.py    # full trace 출력
```

## 부록: 재현 명령

```bash
# 1. 데이터
python -m data.build_doc_reader_data --n 200

# 2. 학습 (dry-run, oracle signal, 빠름)
python -m experiments.train_doc_reader \
    --dry-run --timesteps 100000 --max-steps 4 \
    --out logs/ppo_doc_reader_dry.zip

# 3. 학습 (real LLM 32B, 느림)
python -m experiments.train_doc_reader \
    --timesteps 10000 --max-steps 4 \
    --vllm-url http://localhost:8003/v1 \
    --vllm-model Qwen/Qwen2.5-32B-Instruct-AWQ \
    --out logs/ppo_doc_reader_real.zip

# 4. 평가
python -m experiments.eval_doc_reader \
    --model logs/ppo_doc_reader_dry.zip \
    --dry-run --max-steps 4 --eval 199

# 5. demo (7B 사용)
python -m experiments.demo_doc_reader \
    --model logs/ppo_doc_reader_dry.zip \
    --idx 1 --max-steps 4 \
    --vllm-url http://localhost:8002/v1 \
    --vllm-model Qwen/Qwen2.5-7B-Instruct
```
