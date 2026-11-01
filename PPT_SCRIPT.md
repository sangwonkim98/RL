# 발표 대본 (PPO Router for LLM-Guided Multi-Hop QA)

총 14 슬라이드, 약 12-15분 분량.

---

## 슬라이드 1: Title

> 안녕하세요. 김상원입니다.
> 오늘 발표 주제는 **PPO Router for LLM-Guided Multi-Hop QA**입니다.
> 한 문장으로 요약하면, **멀티홉 질의응답에서 어떤 문서를 읽고 언제 답할지를, 강화학습이 결정하고, LLM은 글 읽기와 답 생성만 담당하는 hybrid 시스템**입니다.
> HotpotQA 데이터셋 distractor 세팅에서, PPO와 vLLM Qwen 7B를 사용했고, 199 episodes에서 100k timesteps 학습해 success 70.9%, 평균 2.11번 read로 결과를 얻었습니다.

---

## 슬라이드 2: Motivation

> 왜 굳이 routing에 RL을 쓰느냐, 가 이 발표의 출발점입니다.
> 기존 IRCoT나 ReAct 같은 접근은 **LLM이 prompt 안에서 직접 '더 검색할지 / 답할지'를 판단**합니다. 문제는 이게 안정적이지 않다는 거예요. 너무 많이 읽거나, 충분치 않은데 답을 내거나, 그 판단 기준이 정량적이지 않다는 거죠.
> 그래서 저희 접근은 routing 결정을 **외부 RL policy에 분리**합니다. PPO는 의사결정—어떤 doc 읽고 언제 멈출지—를 담당하고, LLM은 의미 처리—글 읽기, 답 만들기—만 합니다.
> 중요한 점은 **학습 환경과 추론 환경이 동일**하다는 점이에요. 두 환경 모두 LLM이 env 안에 들어가 있어서, sim-to-real gap이 없습니다.

---

## 슬라이드 3: Dataset

> 데이터는 표준 **HotpotQA distractor split**을 썼습니다.
> 한 질문당 10개 문서가 주어지고, 그 중 2개만 gold, 나머지 8개는 distractor입니다. 2-hop reasoning이 필요한 데이터죠.
> 시간 관계상 전체 7405개 중 199개만 샘플링해서 썼고, bridge 타입 166개, comparison 33개입니다.
> 예시로 들면, '스콧 데릭슨과 에드 우드는 같은 국적인가?'라는 질문에서 gold doc은 두 인물의 위키 문서이고, distractor는 Doctor Strange 같은 관련 영화 문서들입니다. PPO는 텍스트를 직접 보지 않고, doc의 인덱스와 BM25 점수만 봅니다.

---

## 슬라이드 4: Pipeline Overview

> 전체 파이프라인입니다.
> Question과 10개 doc이 들어오면, **DocReaderEnv**가 관찰결과를 PPO에게 줍니다. obs는 32차원 숫자 벡터예요.
> PPO는 11개 액션 중 하나를 고릅니다: **0~9 doc 중 하나 읽기** 또는 **현재까지 읽은 문서 기반 정답 생성** 입니다.
> 읽기를 선택하면, 그 doc 본문이 누적되고 BM25가 재계산됩니다. 그 다음 LLM에게 'evidence가 충분한가?'라고 묻고, 0 또는 1 응답을 받아 obs에 반영합니다.
> answer를 선택하면, LLM에게 question + evidence를 주고 답을 생성시킵니다. 그 답이 gold와 매칭되면 +8, 틀리면 −4 reward.
> **핵심은 LLM이 환경 안에 들어가 있다는 점**입니다. 학습할 때도 똑같이 LLM이 호출됩니다.

---

## 슬라이드 5: RL Formulation

> MDP 정의입니다.
> Action space는 11개. read_doc_0부터 9까지 10개와, answer 1개.
> Observation은 32차원으로, read_mask 10개, bm25 점수 10개, 미독 중 BM25 최고 위치의 one-hot 10개, LLM sufficiency 신호 1개, step normalized 1개입니다.
> Reward는 gold doc 읽으면 +1.5, distractor −0.4, 재읽기 −0.7, answer 정답 +8, 오답 −4, 매 step cost −0.2입니다. step cost는 빨리 끝내라는 유도신호죠.
> 알고리즘은 표준 PPO MlpPolicy, learning rate 5e-4, entropy coefficient 0.03. max_steps는 4로 제한해서 강제 종료시켰고, real Qwen 7B를 env loop에 끼워서 100k timesteps 학습했습니다. 약 53분 걸렸어요.

---

## 슬라이드 6: LLM Calls

> LLM 호출은 딱 **두 종류**만 정의했습니다.
> 첫 번째, **sufficiency check**. 매 read action 후에 한 번 호출됩니다. prompt는 'question + 누적 evidence + Is this sufficient? 0 or 1'. max_tokens 2로 거의 1 토큰만 받아요. 그래서 호출 비용이 매우 낮습니다.
> 두 번째, **answer generation**. PPO가 answer action(10번) 칠 때만 호출됩니다. 한 episode당 최대 1회. prompt는 'question + evidence + 답을 짧게 써라'. max_tokens 32로 짧은 답만 받습니다.
> **인터페이스를 좁게 정의한 게 핵심**입니다. 이 덕분에 한 episode당 LLM이 평균 3-4번 호출되고, 캐시까지 쓰면 학습이 1시간 안에 끝나서, env-in-loop 학습이 실제로 가능했습니다.

---

## 슬라이드 7: Running Example

> 한 episode가 실제로 어떻게 굴러가는지 보겠습니다. 스콧 데릭슨 예시입니다.
> **Step 1**: 아무것도 안 읽은 상태. BM25가 doc#1을 추천합니다. PPO가 doc#1을 읽기로 결정해서 Scott Derrickson 문서를 받습니다. 그 다음 LLM에게 '충분한가?' 물어보고 '0'을 받습니다—한 명만으론 비교 못하니까요.
> **Step 2**: read_mask가 갱신되고, BM25가 재계산됩니다. 이제 누적 evidence에 doc#4 Ed Wood가 점수 위로 올라옵니다. PPO가 doc#4 읽고, LLM이 이번엔 '1'을 줍니다.
> **Step 3**: PPO가 sufficient=1 신호 받고 answer action을 선택합니다. LLM이 'Yes'라고 답하고, gold 'yes'와 매칭되어 success.
> **총 reward +10.4. LLM 호출은 3회. 이게 실제 한 episode 흐름**입니다.

---

## 슬라이드 8: Results

> 이게 핵심 결과입니다. 199 episodes 전체에서, real 7B를 env에 끼고 평가한 진짜 비교입니다.
> 비교 대상은 **Iterative BM25 (no RL)** 입니다. 이건 PPO와 똑같이 행동하지만 RL 없이 그리디하게 매 step BM25 top-1을 읽는 휴리스틱이에요.
> 결과:
> - **Success Rate**: PPO **70.9%** vs BM25 35.2% — **+35.7%p**
> - **Avg Gold Hit**: PPO 1.21 vs BM25 1.26 — 거의 같음
> - **Avg Reads**: PPO **2.11** vs BM25 2.80 — PPO가 **0.69번 덜 읽음**
>
> 흥미로운 건 gold doc 찾는 능력은 비슷한데 success는 두 배라는 거예요. 그리고 더 적게 읽었는데도 더 잘했죠. 이게 RL의 가치입니다.

---

## 슬라이드 9: Training Curves

> 학습 곡선입니다. real 7B를 env loop에 끼고 100k timesteps, 53분 걸렸어요.
> **Episode reward**가 −1.70에서 +5.25로 단조 증가. RL이 정상적으로 학습되고 있다는 신호입니다.
> **Episode length**는 3.43에서 3.12로 감소. **PPO가 점점 일찍 멈추는 걸 학습**한다는 뜻입니다.
> **Entropy**가 감소하면서 policy가 deterministic 해집니다.
> 100k에서 수렴 직전 분위기인데, 더 학습하면 더 나아질 수도 있습니다.

---

## 슬라이드 10: Demo Success

> 성공 사례를 trace로 보겠습니다. idx=5번 episode입니다.
> 질문: '2014 S/S는 어떤 한국 보이그룹의 데뷔 앨범인데, 그 그룹은 누가 만들었나?' 답은 'YG Entertainment'.
> **Step 1**: BM25 top이 doc#2 (2014 S/S). PPO가 읽고 gold hit, reward +1.3.
> **Step 2**: BM25 갱신되고 doc#4 (Winner 밴드)가 위로. PPO가 읽고 또 gold hit. LLM이 sufficient=1 신호 줌.
> **Step 3**: PPO가 answer 결정. LLM이 'YG Entertainment'라고 답. 정답 매칭, reward +7.8.
>
> **총 +10.4, 3 step만에 종료**. 이상적인 경우입니다.

---

## 슬라이드 11: Demo Failure

> 반대로 실패 사례입니다. 스콧 데릭슨 예시인데, 사실 이 episode가 PPO에게 어려운 케이스예요.
> 문제는 **doc#9 Doctor Strange**가 BM25 점수 1위라는 점입니다. 스콧 데릭슨이 감독한 영화라 단어 겹침이 많아서 distractor임에도 BM25가 그걸 가장 관련 있다고 점수 줍니다.
> 그래서 PPO도 그걸 따라서 doc#9, doc#7, doc#5 distractor만 연속 읽다가 max_steps 도달, 답 못 침. 실패.
>
> **이게 시스템의 한계**입니다. BM25 자체가 distractor를 더 높이 점수 주는 hard case에서는 PPO도 흔들립니다. Future work에서 다룰 부분이죠.

---

## 슬라이드 12: What Did PPO Actually Learn

> 그러면 PPO가 정확히 뭘 학습한 건지 분석해봅니다.
> 관찰 1: **Gold hit이 PPO와 BM25 둘 다 비슷**합니다. 1.21 vs 1.26. 둘 다 같은 BM25 신호를 쓰니까 문서 고르는 능력은 비슷한 거죠.
> 관찰 2: 그런데 **success는 PPO가 두 배** (70.9% vs 35.2%).
> 관찰 3: **PPO가 더 적게 읽음** (2.11 vs 2.80).
>
> 종합하면, **PPO가 학습한 건 'gold doc 잘 고르기'가 아니라 '언제 멈춰 답할지' 라는 것**입니다.
> Iterative BM25는 LLM sufficient=1 신호가 오면 그대로 따라가는 수동적 policy인데, PPO는 read_mask, BM25 추세, sufficient 신호를 종합해서 능동적으로 멈출 시점을 결정합니다.
> 결과적으로 불필요한 distractor 읽기를 회피해서, LLM이 noisy하지 않은 evidence로 답을 생성할 수 있게 만들어줍니다.

---

## 슬라이드 13: Limitations

> 솔직히 인정해야 할 한계 다섯 가지입니다.
> 첫째, **데이터 규모**. 199개만 사용했어요. 전체 7405개에서의 일반화는 검증 못했습니다.
> 둘째, **2-hop 한정**. 3-hop 이상에서 어떨지는 미확인.
> 셋째, **feature engineering 의존**. bm25_unread_argmax 같은 강한 hint가 학습에 결정적이었습니다. 별도 ablation에서 이걸 빼면 성능이 폭락하는 걸 확인했어요. PPO가 raw text로부터 처음부터 다 배운 건 아닙니다.
> 넷째, **hard case에 약함**. 방금 보신 Doctor Strange 케이스처럼 BM25가 잘못 점수 주면 PPO도 같이 망합니다.
> 다섯째, **train/test split 분리 안 함**. 같은 199개로 학습하고 평가했습니다.

---

## 슬라이드 14: Summary & Future Work

> 정리하겠습니다.
> PPO Router에 LLM을 env loop에 끼운 hybrid 파이프라인을 만들었고, 학습/추론 환경 일치를 달성했습니다.
> 결과는 **Iterative BM25 baseline 대비 success +35.7%p, 평균 read 횟수는 0.69번 적음**. 같은 BM25 신호와 같은 LLM 위에서, RL이 '언제 멈출지'를 학습한 게 실질적 이득을 만들어냈습니다.
>
> Future work으로는 train/test 분리, 3-hop 이상 데이터, BM25가 약한 full wiki retrieval 환경, 그리고 distractor가 BM25 top 차지하는 hard case에 robust한 정책 학습을 생각하고 있습니다.
>
> 발표 들어주셔서 감사합니다. 질문 받겠습니다.

---

# 예상 질문 대응

### Q: 왜 PPO만 썼나요? DQN이나 A2C는 비교 안 했나요?
> 시간 제약상 한 알고리즘만 깊게 했습니다. PPO가 sample efficiency가 좋고 stable-baselines3에서 검증된 구현이라 선택했습니다. 다른 알고리즘 비교는 future work입니다.

### Q: 199개로 학습/평가가 같은 게 문제 아닌가요?
> 맞습니다, limitations에 적은 그대로입니다. 시간 안에 결과 내려고 그렇게 했고, 진짜 일반화 측정은 held-out split이 필요합니다.

### Q: PPO가 진짜 학습한 건지, 그냥 BM25 따라간 건지 어떻게 보장하나요?
> Iterative BM25가 PPO와 같은 BM25 신호 + 같은 LLM을 쓰는데도 success가 절반밖에 안 됩니다. 그 차이가 RL이 추가로 학습한 부분이라고 봅니다.

### Q: bm25_argmax feature가 없으면 학습이 안 되는데, 이게 진짜 RL이라고 할 수 있나요?
> Feature engineering 의존성은 인정합니다. PPO가 raw BM25 score를 보고 argmax 위치 추출까지 학습하지는 못했습니다. 더 큰 데이터와 더 깊은 네트워크면 가능할 수도 있는데, 이번 setup으론 이 strong hint가 필요했습니다.

### Q: real LLM in loop 학습이 실제로 의미가 있나요? offline pre-trained LLM도 충분하지 않나요?
> 학습/추론 환경 일치가 핵심이었습니다. 만약 학습은 oracle signal로 하고 추론만 real LLM이면, train-infer gap이 생깁니다. 같은 LLM 위에서 학습하면 PPO가 그 LLM의 노이즈/특성에 맞춰 robust policy를 학습할 수 있습니다.

### Q: 이 결과가 더 큰 데이터에서도 유지될 거라 보장할 수 있나요?
> 보장은 못합니다. 199개에서 본 패턴이 일반적인지는 확장 실험이 필요합니다. 다만 BM25 휴리스틱 대비 RL이 추가 이득을 만든다는 정성적 결론은 generalize될 가능성이 있다고 봅니다.

### Q: PPO 대신 simple supervised learning (BC, behavior cloning) 으로도 같은 효과 낼 수 있지 않나요?
> 가능합니다. gold action sequence가 있다면 BC가 더 효율적일 수 있어요. 하지만 우리 setup에는 명시적 gold trajectory가 없고 reward만 있어서 RL이 적합했습니다. BC 비교는 future work입니다.
