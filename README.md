# PPO Router for LLM-Guided Multi-Hop QA

> 상태: 보류 | 기간: 2026-04-06 ~ 2026-05-26 (git 첫 커밋과 마지막 파일 수정일 기준) | 공개: private
> 결과: 강화학습 수업 프로젝트로 코드와 발표 자료까지 만들었다. HotpotQA distractor 199개 episode에서 PPO Router(실제 Qwen2.5-7B 신호로 학습·평가)의 success 70.9%, Iterative BM25 35.2% (`logs/final_compare.json`). 학습과 평가에 같은 199개 episode를 썼고 held-out 평가는 없다. 수치는 UNVERIFIED
> 다음 단계: 없음. 공개 저장소 `RL`(합성 데이터 MVP, 2커밋)을 이 버전으로 바꿀지는 따로 정한다

## 무엇인가

멀티홉 QA에서 어떤 문서를 읽고 언제 답할지를 PPO policy가 정하고, LLM은 근거가 충분한지 판단하는 신호와 최종 답 생성만 맡는 구조다. HotpotQA distractor 설정(질문당 문서 10개, gold 2개)에서 실험했다. 학습 환경 안에 LLM을 넣어 학습과 추론이 같은 환경에서 돈다.

- Action 11개: `read_doc_0`~`read_doc_9`, `answer`
- Observation 32차원: read mask(10), BM25 점수(10), 아직 안 읽은 BM25 최고 문서(10), LLM sufficiency(1), step 비율(1)
- Reward: gold 문서 +1.5, distractor -0.4, 다시 읽기 -0.7, 정답 +8, 오답 -4, step마다 -0.2

## 결과

`logs/final_compare.json` (2026-05-26, n=199, Qwen2.5-7B-Instruct):

| 방법 | success | avg_gold_hit | avg_reads |
|---|---:|---:|---:|
| Iterative BM25 (real 7B) | 35.2% | 1.26/2 | 2.80 |
| PPO Router (real 7B로 100k step 학습, real 7B로 평가) | 70.9% | 1.21/2 | 2.11 |

이전 README에 있던 dry-run 비교(oracle 신호, n=199, max_steps=4):

| 방법 | success | avg_gold_hit | avg_reads |
|---|---:|---:|---:|
| Random read 2 | 2.5% | 0.34/2 | 2.0 |
| BM25 top-2 + ans | 35.2% | 1.29/2 | 2.0 |
| BM25 top-3 + ans | 54.8% | 1.50/2 | 3.0 |
| BM25 top-4 + ans | 63.3% | 1.60/2 | 4.0 |
| PPO Router (dry-run 100k) | 54.8% | 1.69/2 | 3.63 |
| PPO Router (real LLM 10k) | 36.0% | 0.36/2 | 1.56 |

한계:

- `experiments/train_doc_reader.py`와 `experiments/final_compare.py`가 둘 다 `data/hotpot_doc_reader.json`을 읽는다. 학습 episode와 평가 episode가 겹친다.
- 데이터는 HotpotQA dev distractor 7,405개 중 199개(bridge 166, comparison 33)만 썼다 (`PPT_SCRIPT.md`).
- seed 반복이나 신뢰구간은 없다.

## 폴더 안내

| 경로 | 내용 |
|---|---|
| `envs/doc_reader_env.py` | 11-action MDP. LLM 호출이 환경 안에 있다 |
| `data/build_doc_reader_data.py` | HotpotQA를 doc reader 형식으로 바꾼다 |
| `data/hotpot_doc_reader.json` | 위 스크립트로 만든 199개 episode |
| `experiments/train_doc_reader.py`, `eval_doc_reader.py`, `final_compare.py` | 학습, 평가, 최종 비교 |
| `experiments/iterative_bm25_baseline.py` | Iterative BM25 기준선 |
| `logs/` | 비교 결과 JSON, 학습 곡선 그림, TensorBoard 로그, 학습 로그 |
| `PPT_NOTES.md`, `PPT_SCRIPT.md`, `make_ppt.py` | 수업 발표 자료 정리, 대본, 슬라이드 생성 스크립트 |
| `env.py`, `train.py`, `eval.py`, `baselines.py`, `data_generator.py` | 첫 버전(합성 데이터 MVP, 2026-04-06 커밋) |

## 재현

```bash
python -m data.build_doc_reader_data --n 200
python -m experiments.train_doc_reader --dry-run --timesteps 100000 --max-steps 4 --out logs/ppo_doc_reader_dry.zip
python -m experiments.eval_doc_reader --model logs/ppo_doc_reader_dry.zip --dry-run --max-steps 4 --eval 199
```

실제 LLM 신호를 쓰려면 vLLM 서버가 필요하다(`--vllm-url`, `--vllm-model`). 원래 실험은 Qwen2.5-7B-Instruct(:8002)와 Qwen2.5-32B-Instruct-AWQ(:8003)를 썼다.

## 들어 있지 않은 것

- HotpotQA 원본 `data/hotpot_dev_distractor.json`(46M). 공식 배포처에서 받는다.
- 학습된 PPO 모델 `*.zip`. 위 명령으로 다시 만든다.
- 수업 제출용 PPTX 2개. 학번이 든 파일 이름이라 뺐다.
