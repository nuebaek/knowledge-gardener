# 🗓️ week-09-TIL — 모델을 가볍게, 그리고 내 코드를 깨끗하게

> 양자화(Quantization)로 LLM의 무게를 줄이는 원리부터 PEFT/LoRA로 가볍게 미세조정하는 법, 그리고 주차 막판 Alex의 코드 리뷰를 통해 "잘 작동하는 코드"와 "좋은 코드"의 차이를 체감한 주차

**기간:** 2026-07-06 ~ 2026-07-10

**키워드:** `Quantization` `FP32/FP16/BF16` `INT8/INT4` `PTQ/QAT` `GGUF` `PEFT` `LoRA` `QLoRA` `Distillation` `GRPO` `코드 리뷰`

---

## 📋 이번 주 학습 지도

| 날짜        | 주제                                 | 핵심 개념                                                                                                                                                                  |
| --------- | ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 07/06 (월) | LLM Optimization 개론 & Quantization | 3축(데이터타입/PEFT/배포형식), FP32/FP16/BF16, Mixed Precision, Affine Quantization, Calibration, PTQ/QAT, Weight/Activation Quantization, W8A8/W4A16, Outlier Problem, GPTQ/AWQ |
| 07/07 (화) | Quantization 평가 & PEFT 기초          | Quantization Evaluation, GGUF, Llama.cpp, PEFT, LoRA, QLoRA, Adapter Merge, Model Merging                                                                              |
| 07/08 (수) | Finetuning 전략 & 강화학습 기초            | Full vs Partial Finetuning, Unsloth, Fine-Tuning vs RAG, Distillation, RL, GRPO                                                                                        |


---

## 📖 학습 내용

### Day 1 (07/06) — LLM Optimization 개론 & Quantization

**한 줄 요약:** LLM을 실전 배포 가능하게 만드는 최적화를 "데이터타입·PEFT·배포형식"이라는 3축으로 정리하고, 그중 첫 축인 양자화(숫자를 더 적은 비트로 표현하는 것)의 원리를 파고든 날.

**개념 흐름:**

```
LLM Optimization 개론: 3축(데이터타입 / PEFT / 배포형식)
    ↓ (첫 번째 축 — 데이터 타입부터)
FP32 → FP16/BF16 → Mixed Precision Training
    ↓ (더 줄이면?)
INT8/INT4 Quantization
    ↓ (실수를 정수로 어떻게 바꾸나?)
Affine Quantization + Calibration
    ↓ (언제 양자화하나?)
PTQ(학습 후) vs QAT(학습 중 시뮬레이션)
    ↓ (뭘 양자화하나?)
Weight Quantization vs Activation Quantization (W8A8/W4A16)
    ↓ (문제는?)
Outlier Problem → GPTQ, AWQ로 대응
```

---

#### 1-1. LLM Optimization 개론 (3축)

> LLM을 실서비스에 배포 가능하게 만드는 최적화는 크게 세 축으로 나뉜다.

**핵심**
1. **데이터 타입 축**: 파라미터를 몇 비트 실수/정수로 표현할지 (FP32→FP16→INT8→INT4, 양자화)
2. **PEFT 축**: 전체 파라미터를 다 학습하지 않고 일부만 효율적으로 미세조정 (LoRA 등)
3. **배포 형식 축**: 학습된 모델을 실제 추론 환경에 맞게 변환·경량화하는 포맷 (GGUF 등)

이 세 축은 서로 독립적이면서도 결합 가능 — 예: LoRA로 미세조정한 뒤 양자화해서 GGUF로 배포.

---

#### 1-2. FP32, FP16, BF16, Mixed Precision Training

> 부동소수점 표현 비트 수를 줄여 메모리와 연산량을 아끼는 첫 단계.

**핵심**
- **FP32**: 32비트 단정밀도, 표현 범위·정밀도 다 좋지만 메모리를 가장 많이 씀 (전통적 학습 기본값)
- **FP16**: 16비트 반정밀도. 메모리 절반이지만 표현 가능한 지수(exponent) 범위가 좁아 **오버플로우/언더플로우**에 취약
- **BF16**: 16비트지만 FP32와 지수부 비트 수가 같아 표현 범위는 FP32와 동일하고 가수부(정밀도)만 줄임 → 학습 안정성이 FP16보다 좋아 최근 LLM 학습 표준
- **Mixed Precision Training**: 가중치는 FP32 마스터 사본으로 정밀하게 유지하면서, 순전파·역전파 연산 자체는 FP16/BF16으로 빠르게 수행해 속도와 정확도를 절충

---

#### 1-3. INT8/INT4 Quantization과 Affine Quantization

> 정수(Integer) 몇 비트로 압축해 모델 크기를 더 줄이는 것 — FP16→INT8은 저장공간 추가로 절반, INT4는 다시 절반.

**핵심 — Affine Quantization**
- 실수 범위 [min, max]를 정수 범위(예: INT8이면 -128~127)로 선형 매핑하는 방식
- 공식 핵심 요소: **Scale(축척)**과 **Zero-point(영점)** — `실수값 ≈ scale × (정수값 - zero_point)`
- **Calibration(보정)**: 실제 데이터를 흘려보며 각 레이어의 실숫값 분포(min/max 등)를 관찰해 최적의 scale/zero-point를 결정하는 과정

---

#### 1-4. PTQ vs QAT

| | PTQ (Post-Training Quantization) | QAT (Quantization-Aware Training) |
|---|---|---|
| 시점 | 학습이 끝난 모델에 사후 적용 | 학습 과정 중에 양자화를 시뮬레이션하며 학습 |
| 비용 | 저렴, 빠름 (재학습 불필요) | 비쌈 (재학습 필요) |
| 정확도 | 상대적으로 손실 가능 | 양자화로 인한 손실을 모델이 학습으로 보정 → 정확도 우수 |

**Static vs Dynamic Quantization** (PTQ 하위 구분)
- **Static**: 가중치와 활성화값 모두 사전에 Calibration으로 scale 고정
- **Dynamic**: 가중치는 미리 양자화, 활성화값은 추론 시점에 실시간으로 scale 계산 — Calibration 데이터가 없어도 되지만 연산 오버헤드가 있음

---

#### 1-5. Weight Quantization vs Activation Quantization, W8A8/W4A16

**핵심**
- **Weight Quantization**: 모델 파라미터(가중치) 자체를 양자화 — 저장 공간 절감이 핵심 목표
- **Weight-only Quantization**: 가중치만 양자화하고 활성화값(중간 연산 결과)은 원래 정밀도(FP16 등) 유지 — 구현이 단순하고 정확도 손실이 적어 널리 쓰임
- **Activation Quantization**: 레이어를 통과하며 계속 값이 바뀌는 활성화값까지 양자화 — 연산 속도까지 확보하지만 난이도 ↑
- **표기법**: `W4A16` = 가중치 4비트 + 활성화 16비트(weight-only) / `W8A8` = 가중치·활성화 모두 8비트
- **KV Cache Quantization**: 긴 컨텍스트에서 메모리를 많이 잡아먹는 KV 캐시도 양자화 대상이 됨

---

#### 1-6. Outlier Problem, GPTQ, AWQ

> 양자화의 최대 난관은 일부 레이어에 비정상적으로 큰 값(Outlier)이 섞여 있어, 이 값에 scale을 맞추면 나머지 대다수 값의 정밀도가 뭉개진다는 것.

**핵심**
- **Outlier Problem**: 소수의 극단값 때문에 전체 양자화 품질이 떨어지는 문제
- **GPTQ**: 레이어별로 가중치를 순차적으로 양자화하면서, 양자화로 생긴 오차를 다음 가중치에 보정해 반영하는 2차 근사 기반 기법
- **AWQ (Activation-aware Weight Quantization)**: 모든 가중치를 동일하게 취급하지 않고, 활성화값 크기를 기준으로 "중요한" 가중치 채널을 찾아 그 채널만 더 정밀하게 보존

---

### Day 2 (07/07) — Quantization 평가 & PEFT 기초

**한 줄 요약:** 양자화가 잘 됐는지 평가하는 법과 실제 배포 포맷(GGUF), 그리고 파라미터 효율적 미세조정의 대표주자 LoRA/QLoRA를 배운 날.

**개념 흐름:**

```
양자화했다 — 근데 성능이 얼마나 떨어졌나?
    ↓
Quantization Evaluation
    ↓ (실제 배포는 어떤 포맷으로?)
GGUF & Llama.cpp
    ↓ (전체 파라미터 재학습 말고, 효율적으로 미세조정하려면?)
PEFT 개념
    ↓ (대표 기법은?)
LoRA → QLoRA(양자화 + LoRA 결합)
    ↓ (학습한 어댑터를 실전에 쓰려면?)
Adapter Merge & Model Merging
```

---

#### 2-1. Quantization Evaluation

> 양자화 전후 모델의 성능 저하를 정량적으로 측정하는 과정.

**핵심**
- 단순히 "모델이 작아졌다"가 목표가 아니라, 벤치마크 점수/perplexity(혼란도) 등으로 원본 모델 대비 성능 저하 폭을 확인해야 실사용 가능 여부를 판단할 수 있음
- 비트를 낮출수록 크기·속도는 좋아지지만 성능 저하 트레이드오프가 커짐 — 어느 지점에서 타협할지가 실무적 의사결정

---

#### 2-2. GGUF와 Llama.cpp

> GGUF = 양자화된 LLM을 CPU/저사양 환경에서도 효율적으로 구동하기 위한 단일 파일 배포 포맷. Llama.cpp = 이 포맷을 읽어 C/C++로 빠르게 추론하는 경량 런타임.

**핵심**
- 파이썬/CUDA 없이도 노트북 CPU 등에서 로컬 LLM 구동을 가능하게 한 핵심 생태계
- 다양한 양자화 레벨(Q4_K_M 등 표기)을 하나의 포맷 안에서 선택적으로 지원

---

#### 2-3. PEFT (Parameter-Efficient Fine-Tuning)

> 사전학습된 거대 모델의 전체 파라미터를 다 건드리지 않고, 극히 일부(또는 추가된 소수 파라미터)만 학습해 미세조정하는 기법군.

**왜 쓰는가**
- Full Finetuning은 수십~수백억 파라미터를 전부 업데이트해야 해서 GPU 메모리·비용이 막대함 → 개인/소규모 팀은 사실상 불가능
- PEFT는 원본 가중치는 고정(freeze)하고 훨씬 적은 파라미터만 학습해 비슷한 효과를 훨씬 저렴하게 달성

---

#### 2-4. LoRA와 QLoRA

> **LoRA(Low-Rank Adaptation)**: 원본 가중치 행렬은 고정한 채, 그 옆에 훨씬 작은 저랭크(rank) 행렬 쌍(A, B)을 추가해 그 소수 파라미터만 학습하는 기법.

**핵심**
- 원본 가중치 W는 동결, 변화량을 `ΔW = A·B` (A, B는 저랭크 행렬)로 근사해 A와 B만 학습
- 추론 시 `W + ΔW`로 합쳐 쓰거나, 어댑터를 분리한 채로도 사용 가능 — 원본 모델은 그대로 두고 작은 어댑터 파일만 여러 개 갈아끼울 수 있음

**QLoRA**
- 원본 모델을 먼저 4비트로 양자화해 메모리를 극단적으로 줄인 뒤, 그 위에 LoRA 어댑터를 얹어 학습 — 양자화(축의 1번)와 PEFT(축의 2번)를 결합한 실전 기법
- 소비자용 GPU 한 장으로도 대형 모델 미세조정이 가능해진 핵심 돌파구

---

#### 2-5. Adapter Merge와 Model Merging

**핵심**
- **Adapter Merge**: 학습된 LoRA 어댑터(A·B)를 원본 가중치에 수학적으로 합쳐(`W_new = W + A·B`) 별도 어댑터 없이 단일 모델로 만드는 과정 — 배포 단순화, 추론 시 추가 연산 없음
- **Model Merging**: 서로 다른 목적으로 파인튜닝된 여러 모델(또는 어댑터)의 가중치를 가중 평균 등으로 합쳐 하나의 모델에 여러 능력을 동시에 담으려는 시도

---

### Day 3 (07/08) — Finetuning 전략 & 강화학습 기초

**한 줄 요약:** Full Finetuning과 PEFT의 실전 선택 기준, RAG와 파인튜닝의 역할 차이, 그리고 지식 증류·강화학습(GRPO)까지 미세조정 생태계를 넓게 훑은 날.

**개념 흐름:**

```
Full Finetuning vs Partial Finetuning(PEFT) 선택 기준
    ↓ (실전에서 더 쉽게 하려면?)
Unsloth (파인튜닝 가속 라이브러리)
    ↓ (근데 파인튜닝이 항상 답인가?)
Fine-Tuning vs RAG — 언제 뭘 쓰나
    ↓ (큰 모델의 지식을 작은 모델로 옮기려면?)
Distillation (지식 증류)
    ↓ (사람 피드백/보상으로 모델을 개선하려면?)
강화학습(RL) 개념 → GRPO
```

---

#### 3-1. Full Finetuning vs Partial Finetuning (PEFT)

| | Full Finetuning | Partial Finetuning (PEFT) |
|---|---|---|
| 학습 대상 | 전체 파라미터 | 일부 파라미터(또는 추가된 소수 파라미터) |
| 비용 | 매우 높음 (대형 GPU 클러스터) | 훨씬 저렴 (GPU 한두 장도 가능) |
| 성능 상한 | 이론상 더 높은 상한 | 대부분의 실무 태스크에서 충분한 근사 |
| 언제 쓰나 | 도메인이 완전히 다르고 자원이 충분할 때 | 대부분의 실무 상황 (기본 선택지) |

---

#### 3-2. Unsloth

> LoRA/QLoRA 기반 파인튜닝을 커널 최적화로 대폭 가속화(속도·메모리 효율 개선)하는 오픈소스 라이브러리.

**핵심**
- 표준 Hugging Face 파인튜닝 스택 대비 학습 속도와 메모리 사용량을 크게 개선 — 개인 GPU 환경에서 파인튜닝 실습 접근성을 높여줌

---

#### 3-3. Fine-Tuning vs RAG

> 둘은 경쟁 관계가 아니라 해결하는 문제가 다르다.

**핵심**
- **Fine-Tuning**: 모델의 "행동 방식·말투·형식·추론 패턴"을 바꾸는 데 적합. 최신 정보를 넣는 데는 비효율적(재학습 필요, 환각 위험)
- **RAG**: 모델의 파라미터는 그대로 두고 "최신 지식·외부 정보"를 검색으로 주입하는 데 적합. 모델의 근본적인 행동 패턴 자체를 바꾸지는 못함
- 실무에서는 "행동을 바꿔야 하나(FT) vs 정보를 알려줘야 하나(RAG)"를 먼저 구분해야 함 — 둘을 함께 쓰는 것도 일반적

---

#### 3-4. Distillation (지식 증류)

> 크고 성능 좋은 Teacher 모델의 출력(또는 확률 분포)을 작은 Student 모델이 모방하도록 학습시켜, 작은 모델이 큰 모델의 능력을 일부 이어받게 하는 기법.

**핵심**
- Student는 정답 레이블뿐 아니라 Teacher의 **soft label(확률 분포)**까지 학습 신호로 사용 — 단순 정답보다 더 풍부한 정보 전달
- 경량화된 배포 모델을 만들 때 양자화·PEFT와 함께 쓰이는 또 다른 축

---

#### 3-5. 강화학습(RL)과 GRPO

> RL = 모델이 행동(출력)을 하고 그에 대한 보상(reward)을 받아, 보상을 최대화하는 방향으로 정책을 개선하는 학습 패러다임. LLM 정렬(alignment)의 핵심 축.

**핵심 — GRPO (Group Relative Policy Optimization)**
- 동일한 프롬프트에 대해 여러 개의 응답을 그룹으로 생성한 뒤, 그룹 내 상대적 우열(그룹 평균 대비 잘했는지)로 보상을 정규화해 정책을 업데이트
- 별도의 가치 함수(critic) 모델 없이도 상대적 비교만으로 학습 신호를 얻을 수 있어 기존 RLHF(PPO 기반) 대비 구조가 단순하고 자원 효율적

---

## 🔗 이번 주 개념 흐름

```
[월 07/06] Quantization 원리
    3축 개론 → FP32/FP16/BF16 → Affine Quantization/Calibration
    → PTQ/QAT → Weight/Activation Quantization(W8A8/W4A16) → Outlier(GPTQ/AWQ)
    ↓ (얼마나 잘 줄였나?)
[화 07/07] Quantization 평가 & PEFT
    Evaluation → GGUF/Llama.cpp → PEFT → LoRA → QLoRA → Adapter/Model Merging
    ↓ (미세조정을 언제 어떻게 쓰나?)
[수 07/08] Finetuning 전략 & RL
    Full vs Partial → Unsloth → Fine-Tuning vs RAG → Distillation → RL/GRPO
    ↓ (내가 짠 코드는 실제로 괜찮은가?)
```

이번 주는 "모델을 어떻게 가볍게 만드나(양자화)"에서 "모델을 어떻게 효율적으로 바꾸나(PEFT/RL)"로 이어지다가, 목요일에 갑자기 "내가 짜는 코드 자체는 괜찮은가"로 시점이 전환된 게 특징적이었다.

---