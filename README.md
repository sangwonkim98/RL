# PPO Router for LLM-Guided Multi-Hop QA

멀티홉 QA에서 어떤 문서를 읽고 언제 답할지를 PPO policy가 정하고, LLM은 근거 충분성 판단과 답 생성만 맡는 강화학습 수업 프로젝트다.

> 상태: 종료 | 기간: 2026-04-06 ~ 2026-05-26 (git 첫 커밋과 마지막 파일 수정일 기준) | 공개: private
> 결과: HotpotQA distractor 199개 episode에서 PPO Router를 실제 Qwen2.5-7B 신호로 학습·평가하고, Iterative BM25 기준선과 비교했다. 수업 발표 자료(대본, 슬라이드 생성 스크립트)까지 만들었다. 학습과 평가에 같은 episode를 써서 held-out 결과는 없다.
> 다음 단계: 없음. 공개 저장소 `RL`(합성 데이터로 만든 첫 버전)을 이 버전으로 바꿀지는 따로 정한다.

## 무엇·왜

IRCoT 같은 LLM 멀티홉 추론은 LLM이 prompt 안에서 "더 찾을지, 지금 답할지"를 직접 정한다. 이 판단은 자주 흔들린다. 문서를 너무 많이 읽거나 너무 일찍 멈춘다.

이 프로젝트는 그 결정을 바깥의 RL policy에 맡긴다. PPO가 읽을 문서와 멈출 시점을 고르고, LLM은 지금 근거로 답할 수 있는지(0/1)와 최종 답만 낸다. LLM 호출을 학습 환경 안에 넣었기 때문에 학습과 추론이 같은 환경에서 돈다.

실험은 HotpotQA distractor 설정(질문당 문서 10개, 그중 gold 2개)에서 했다.

- Action 11개: `read_doc_0`~`read_doc_9`, `answer`
- Observation 32차원: read mask(10), BM25 점수(10), 아직 안 읽은 문서 중 BM25 최고 문서 one-hot(10), LLM sufficiency(1), step 비율(1)
- Reward: gold 문서 +1.5, distractor -0.4, 다시 읽기 -0.7, 정답 +8, 오답 -4, step마다 -0.2

위 값은 `envs/doc_reader_env.py`의 코드 기준이다. 같은 파일 맨 위 docstring에는 더 이른 설정(22차원, 보상 +1.0/-0.3 등)이 남아 있다.

## 어떻게 쓰나

requirements 파일은 없다. 코드가 import하는 패키지는 `stable-baselines3`, `gymnasium`, `numpy`, `openai`(실제 LLM 호출), `matplotlib`(그림), `python-pptx`(`make_ppt.py`)다. 명령은 저장소 루트에서 실행한다.

```bash
# 1) HotpotQA를 doc reader 형식으로 변환 (원본 data/hotpot_dev_distractor.json 필요)
python -m data.build_doc_reader_data --n 200

# 2) LLM 없이 oracle 신호로 학습과 평가 (dry-run)
python -m experiments.train_doc_reader --dry-run --timesteps 100000 --max-steps 4 --out logs/ppo_doc_reader_dry.zip
python -m experiments.eval_doc_reader --model logs/ppo_doc_reader_dry.zip --dry-run --max-steps 4 --eval 199

# 3) 실제 LLM 신호로 학습, 기준선, 최종 비교 (vLLM 서버 필요)
python -m experiments.train_doc_reader --timesteps 100000 --max-steps 4 \
  --vllm-url http://localhost:8002/v1 --vllm-model Qwen/Qwen2.5-7B-Instruct \
  --out logs/ppo_real_7b_100k.zip
python -m experiments.iterative_bm25_baseline --n 199 --max-steps 4
python -m experiments.final_compare --ppo-model logs/ppo_real_7b_100k.zip --n 199 --out logs/final_compare.json
```

- 변환된 199개 episode(`data/hotpot_doc_reader.json`)는 저장소에 들어 있다. 1단계는 다시 만들 때만 필요하다.
- dry-run은 "gold 2개를 다 읽으면 충분, 그때 정답" 규칙으로 LLM을 대신한다. 환경과 학습 코드 점검용이다.
- 실제 LLM 모드는 OpenAI 호환 vLLM 서버가 필요하다(`--vllm-url`, `--vllm-model`). 원래 실험은 GPU 서버에서 Qwen2.5-7B-Instruct(:8002)와 Qwen2.5-32B-Instruct-AWQ(:8003)를 띄워 썼다. API 키는 쓰지 않는다.
- 결과 수치는 `logs/final_compare.json`, 학습 곡선은 `logs/training_curves_real_7b.png`에 있다.

## 구조

| 경로 | 내용 |
|---|---|
| `envs/doc_reader_env.py` | 최종 11-action MDP. LLM 호출이 환경 안에 있다 |
| `envs/doc_reader_env_no_argmax.py` | BM25 최고 문서 feature를 뺀 ablation 환경 |
| `envs/multihop_nav_env.py`, `envs/path_router_env.py` | 중간 단계에서 시험한 환경 |
| `data/build_doc_reader_data.py` | HotpotQA를 doc reader 형식으로 바꾼다 |
| `data/hotpot_doc_reader.json` | 위 스크립트로 만든 199개 episode |
| `data/multihop_generator.py`, `data/build_path_data.py` | 중간 단계 환경용 데이터 생성 |
| `experiments/train_doc_reader.py`, `eval_doc_reader.py`, `final_compare.py` | 최종 환경의 학습, 평가, 최종 비교 |
| `experiments/iterative_bm25_baseline.py` | Iterative BM25 기준선 |
| `experiments/train_doc_reader_no_argmax.py` | ablation 학습 |
| `experiments/demo_doc_reader.py`, `demo_ircot_prompt.py` | 발표용 rollout 예시, IRCoT식 prompt 예시 |
| `experiments/train_ppo.py`, `train_router.py`, `eval_router.py`, `sanity_env_check.py` | 중간 단계 환경의 학습, 평가, 점검 |
| `logs/` | 비교 결과 JSON, 학습 곡선과 기준선 그림, TensorBoard 로그(`tb_doc/`), 학습 로그 |
| `PPT_NOTES.md`, `PPT_SCRIPT.md`, `make_ppt.py` | 발표 자료 정리, 대본과 예상 질문, 슬라이드 생성 스크립트 |
| `env.py`, `train.py`, `eval.py`, `baselines.py`, `data_generator.py`, `plot_results.py`, `preprocess_hotpot.py` | 첫 버전(합성 데이터 MVP, 2026-04-06 커밋) |
| `data.json`, `hotpot_processed.json`, `metrics.json`, `evaluation_results.png`, `ppo_doc_nav_model.zip` | 첫 버전의 데이터, 결과, 모델 |

첫 버전 스크립트는 `ppo_navigation` 패키지로 import한다(`from ppo_navigation.env import DocNavEnv`). 이 폴더 이름이 `ppo_navigation`일 때를 가정한 코드라 지금 구조에서는 바로 돌지 않는다.

## 세부 기록

### 실험 조건과 한계

- 데이터는 HotpotQA dev distractor 7,405개 중 199개(bridge 166, comparison 33)만 썼다(`PPT_SCRIPT.md`).
- `experiments/train_doc_reader.py`와 `experiments/final_compare.py`가 둘 다 `data/hotpot_doc_reader.json`을 읽는다. 학습 episode와 평가 episode가 겹친다.
- seed 반복이나 신뢰구간은 없다.
- 최종 비교는 실제 Qwen2.5-7B-Instruct 신호로 100k step 학습한 모델(`logs/ppo_real_7b_100k.zip`)과 같은 7B를 쓴 Iterative BM25를 199개 episode에서 비교했다. 지표는 success rate, 평균 gold hit, 평균 read 횟수다.
- 이보다 앞서 dry-run(oracle 신호) 비교도 했다. Random read 2, BM25 top-2/3/4 후 답, dry-run 100k PPO, 실제 LLM 10k PPO를 같은 199개 episode에서 비교했다. 표는 이 저장소의 git 기록 중 2026-11-01 커밋의 README에 있다.
- 발표 대본의 한계 슬라이드는 데이터 규모, train/test 미분리 등 다섯 가지를 적었다. BM25가 distractor를 더 높게 매기는 어려운 경우에는 PPO도 흔들렸다.

### 들어 있지 않은 것

- HotpotQA 원본 `data/hotpot_dev_distractor.json`(46M). 공식 배포처에서 받는다.
- `logs/`의 학습된 PPO 모델 `*.zip`. git 제외 규칙(`*.zip`)으로 커밋하지 않았다. 위 명령으로 다시 만든다. 첫 버전 모델 `ppo_doc_nav_model.zip`만 원래 기록대로 들어 있다.
- 수업 제출용 PPTX 2개. 학번이 든 파일이라 뺐다. `make_ppt.py`로 슬라이드를 다시 만들 수 있다.
