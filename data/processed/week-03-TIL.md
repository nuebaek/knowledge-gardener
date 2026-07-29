# 🗓️ week-03-TIL — 데이터 분석 도구 완주: Numpy부터 Seaborn/Scipy까지

> 배열 연산(Numpy)에서 시작해 시각화 도구(Matplotlib → Seaborn)를 한 층씩 쌓고, 중간에 멘토링으로 학습 방향을 재점검한 주차

**기간:** 2026-05-25 ~ 2026-05-29

**키워드:** `Numpy` `Pandas` `Matplotlib` `데이터시각화` `Seaborn` `Scipy` `정규분포` `시계열`

---

## 📋 이번 주 학습 지도

| 날짜 | 주제 | 핵심 개념 |
|------|------|-----------|
| 05/26 (화) | Numpy / Pandas 기초 | 차원(ndim), Shape, Data Type, 인덱싱, 브로드캐스팅, Series/DataFrame, Filtering, Grouping |
| 05/27 (수) | 데이터 시각화 개론 | 정형/비정형/반정형 데이터, Matplotlib, 막대그래프, 히스토그램, 산점도, 박스플롯, 다중그래프, 벤다이어그램 |
| 05/28 (목) | 멘토링 | 진로/학습법 멘토링(GPU 클라우드, 추천시스템, OS, 평가지표, 코딩 연습법) |
| 05/29 (금) | Seaborn + Scipy | 범주형/연속형/관계형/시계열 데이터, Resampling, Moving Average, 정규분포 |

---

## 📖 학습 내용

### Day 2 (05/26) — Numpy / Pandas 기초

#### 2-1. Numpy

> 데이터 분석을 위한 도구를 제공하는 파이썬 라이브러리. 다차원 배열과 행렬 연산을 처리할 수 있다.

**핵심**
- 특징: 배열 객체, 연산 속도, 브로드캐스팅, 수학 함수 제공, 인덱싱 및 슬라이싱, 다른 라이브러리와의 호환

---

#### 2-2. 차원(ndim)과 Shape

> 차원은 배열을 구성하는 축(axis)의 개수, Shape는 배열의 각 차원별 요소 개수를 나타내는 튜플.

**핵심**
- `.ndim`으로 차원 확인, `reshape()`으로 형태 변경
- `np.newaxis`: 배열에 새로운 차원을 추가 — 행벡터 변환은 `vector[:, np.newaxis]`, 열벡터 변환은 `vector[np.newaxis, :]`
- `.shape` 속성으로 각 차원별 요소 개수 확인. `reshape()`, `resize()`, `flatten()`, `ravel()`, `transpose()` 등으로 유연하게 변경·조작
  - `reshape()`: 새 형태의 배열 크기는 원래 배열 크기와 같아야 함
  - `resize()`: 원본 배열 자체를 변경, 필요시 확장·축소
  - `flatten()`: 다차원 → 1차원 변환 (복사본)
  - `ravel()`: 다차원 → 1차원 변환이지만 원본 배열에 대한 참조(view) 반환

| 개념 | 설명 | 예제 |
|---|---|---|
| 차원(ndim) | 배열이 가진 축(axis)의 개수 | `array.ndim → 2` |
| 형태(shape) | 각 차원별 요소 개수를 튜플로 | `array.shape → (2, 3)` |

---

#### 2-3. Data Type

> 배열의 요소는 정수, 부동소수점, 복소수, 문자열, 불리언, 사용자 정의 타입 등 다양한 형식이 존재한다.

**핵심**

| 데이터 타입 | 설명 | 예시 |
|---|---|---|
| 정수형 (Integers) | 부호 있는/없는 정수 | `int8, int16, int32, int64, uint8, uint16` |
| 부동소수점형 (Floating Point) | 소수점 포함 숫자 | `float16, float32, float64` |
| 복소수형 (Complex Numbers) | 실수+허수 | `complex64, complex128` |
| 문자열형 (Strings) | 고정 길이 문자열 | `str_` 또는 `unicode_` |
| 불리언형 (Boolean) | True/False | `bool_` |
| 객체형 (Object) | 파이썬 객체 | `object_` |

**데이터 타입 변환 시 주의할 점**
- 정수 → 부동소수점 변환 시 정밀도 증가, 반대는 소수점 절삭 가능
- 작은 크기로 변환하면 메모리 절약되지만 값의 범위를 벗어나면 오버플로우 발생 가능

| 기능 | 설명 | 예제 코드 |
|---|---|---|
| `astype()` | 명시적 데이터 타입 변환 | `array.astype(np.float32)` |
| `np.int32()` | 정수형으로 변환 | `np.int32(array)` |
| `np.float64()` | 부동소수점형으로 변환 | `np.float64(array)` |

---

#### 2-4. 인덱싱과 연산

**핵심**
- 인덱싱: 정수 인덱스 / 슬라이싱 / 불리언 인덱싱(조건식 결과로 원소 선택) / 팬시 인덱싱

| 연산 유형 | 설명 | 예시 함수 |
|---|---|---|
| 산술 연산 | 배열 요소별 사칙연산 | `+`, `-`, `*`, `/`, `np.add()` |
| 비교 연산 | 요소 간 크기 비교, 불리언 반환 | `>`, `<`, `==`, `np.equal()` |
| 논리 연산 | 불리언 배열 AND/OR | `np.logical_and()`, `np.logical_or()` |
| 통계 연산 | 평균·최댓값·최솟값 등 | `np.mean()`, `np.sum()`, `np.min()`, `np.max()` |
| 선형대수 연산 | 행렬 곱·전치·행렬식 | `np.dot()`, `np.linalg.inv()`, `np.transpose()` |
| 브로드캐스팅 | 크기가 다른 배열 간 연산 지원 | 자동 배열 크기 조정 |

- **브로드캐스팅**: 크기가 다른 배열 간 연산을 가능하게 함 — 작은 배열이 자동으로 확장되어 큰 배열과 연산 수행
- **universal 함수**: 배열의 모든 요소에 동일한 연산을 적용 (산술/삼각/지수로그/비교/논리/비트 연산)

**코드:**
```python
a = np.array([1, 2, 3])
result = np.empty_like(a)  # 결과를 기록할 기존 배열

# np.multiply(): 요소별 곱, out으로 결과 저장
np.multiply(a, 10, out=result)
print("np.multiply(a, 10, out=result):", result)
```

---

#### 2-5. Pandas — Series / DataFrame

**핵심**
- **Series**: 인덱스를 가지는 1차원 배열 형태의 데이터 구조. 기본 속성 — values, index, dtype, shape, size, name
- **DataFrame**: 행과 열로 구성된 2차원 테이블 형태의 데이터 구조

---

#### 2-6. Filtering / Grouping

**Filtering**
> Pandas에서 조건을 적용하여 데이터프레임이나 시리즈에서 특정 행이나 값을 선택하는 과정. 유의미한 정보만 추출하고 데이터 분석 전 정제·필요한 부분만 다룰 수 있게 한다.

- `df[조건]`: 불리언 인덱싱 사용한 필터링 (조건식 결과 True/False 배열로 True 행만 선택), 논리 연산자 `&`, `|`, `~`
- `query`: SQL과 유사한 문자열 표현식으로 조건 작성, 행을 선택하는 메서드
- `isin()`: 여러 값 중 하나라도 포함되어 있는지 검사해 해당 행 선택
- `str.contains()`, `str.startswith()`: 특정 패턴/접두어 포함 여부를 True/False로 반환
- `apply()`

**Grouping**
> 데이터를 특정 기준에 따라 그룹화하여 집계, 변환, 필터링 등의 연산을 수행하는 기능

- `groupby()`

---

### Day 3 (05/27) — 데이터 시각화 개론

#### 3-1. 데이터 시각화

> 데이터를 그래프나 차트와 같은 시각적 형식으로 표현하여 패턴, 추세, 인사이트를 효과적으로 전달하는 과정

**왜 쓰는가**
- 데이터를 보기 좋게 그래프·차트로 표현해 인사이트를 명확하게 전달하기 위해. 협업(커뮤니케이션)에도 유용
- 유의미한 정보 제공에 집중해야 한다 — 예쁜 그래프가 목적이 아니라 인사이트 전달이 목적

**종류**

| 분류 | 데이터 유형 | 표현 방법 |
|---|---|---|
| 양적 데이터 시각화 | 수치형 데이터 | 선 그래프, 막대 그래프, 히스토그램 |
| 질적 데이터 시각화 | 범주형 데이터 | 파이 차트, 트리맵 |
| 시계열 데이터 시각화 | 시계열 데이터 | 선 그래프, 영역 그래프 |
| 지리적 데이터 시각화 | 지리 데이터 | 지도 기반 시각화 |

---

#### 3-2. 정형/비정형/반정형 데이터

> 정형: 표나 데이터베이스처럼 행·열 구조를 가진 체계적 데이터. 비정형: 고정된 구조 없이 텍스트·이미지·영상 등으로 존재하는 데이터.

**핵심**
- 정형 데이터: 고정된 구조, 빠른 검색·처리, 데이터 무결성 보장
- 비정형 데이터: 다양한 형식, 방대한 양, 실시간 증가
- **반정형 데이터**: 일부 구조는 있지만 고정 스키마 없이 유동적 (JSON, XML, YAML, TOML) — 스키마가 데이터와 하나로 붙어 있는 "셀프 스키마"

**비유:** 정형 = 잘 정리된 엑셀 시트, 비정형 = 메모장, 반정형 = 들여쓰기 있는 JSON

**종류**
- 정형: RDBMS, 엑셀, CSV / 반정형: JSON, XML, YAML, TOML / 비정형: 파일 시스템, NoSQL, 클라우드 스토리지

---

#### 3-3. Matplotlib

> 파이썬에서 다양한 차트와 그래프를 생성할 수 있도록 지원하는 시각화 라이브러리

**핵심**
- `pyplot`으로 그래프 생성·출력
- Pandas 내장 시각화(`DataFrame.plot()`)는 Matplotlib 기반이지만 간단한 탐색용, 세부 조정은 Matplotlib이 강력함

| 비교 | Pandas 내장 시각화 | Matplotlib |
|---|---|---|
| 사용법 | `DataFrame.plot(kind="")` 한 줄 | `plt.figure()`, `plt.plot()`, `plt.show()` 단계 필요 |
| 커스터마이징 | 기본 수준 | 세부 요소까지 정밀 조정 가능 |
| 목적 | 빠른 데이터 탐색 | 고급 시각화 및 발표용 차트 제작 |

---

#### 3-4. 막대 그래프 (Bar Chart)

> 범주형 데이터의 크기를 막대의 길이로 표현해 "서로 다른 범주 간 상대적 크기·비율을 빠르게 비교"하는 시각화

**핵심**
- 가로축: 범주(categorical), 세로축: 값(numerical). `.bar()`로 생성
- 막대 그래프 = **범주 비교**, 히스토그램 = **분포 확인** — 헷갈리지 말 것

```python
import matplotlib.pyplot as plt
categories = ['A', 'B', 'C', 'D']
values = [10, 25, 15, 30]
plt.bar(categories, values, color='steelblue')
plt.xlabel("Category"); plt.ylabel("Value"); plt.title("Bar Chart")
plt.show()
```

---

#### 3-5. 히스토그램

> 연속형 데이터를 구간(bin)으로 나누고 각 구간의 빈도수를 막대로 표현해 "데이터의 분포 형태를 파악"하는 시각화

**핵심**
- bin(구간) 수가 중요 — 너무 많으면 패턴을 보기 어렵고, 너무 적으면 특징을 놓침
- AI 학습 시 feature 분포 확인(정규화 필요 여부), 가중치·활성값 분포 모니터링(gradient explosion/vanishing 확인)에도 활용

```python
import matplotlib.pyplot as plt, numpy as np
data = np.random.randn(1000)
plt.hist(data, bins=20, color='skyblue', edgecolor='black')
plt.xlabel("Value"); plt.ylabel("Frequency"); plt.title("Basic Histogram")
plt.show()
```

---

#### 3-6. 산점도 (Scatter Plot)

> 두 변수를 x축·y축에 배치해 점으로 표현, "두 변수 간의 관계·상관성·군집을 시각적으로 파악"하는 시각화

**핵심**
- x, y 모두 수치형이어야 함. 점의 색상/크기로 제3의 변수도 표현 가능(버블 차트)
- 분류 문제에서 클래스 구분 가능성 파악, 예측값 vs 실제값 분포 분석, 이상치 탐지에 활용

---

#### 3-7. 박스 플롯 (Box Plot)

> 데이터의 분포(중앙값, 사분위수, 이상치)를 한눈에 요약해 "전체 분포와 이상치를 동시에 비교"하는 시각화

**핵심**
- 구성: 박스(Q1~Q3), 중앙선(중앙값), 수염(IQR × 1.5), 점(이상치). IQR = Q3 - Q1
- 히스토그램보다 간결하게 분포를 요약, 여러 그룹을 나란히 비교할 때 강력

---

#### 3-8. 고급 다중 그래프

> 하나의 Figure 안에 여러 Axes(서브플롯)를 배치해 "서로 다른 데이터나 지표를 동시에 비교·분석"하는 기법

**핵심**
- `plt.subplots(nrows, ncols)`로 여러 axes 생성, `axes[i]`로 개별 접근
- `plt.tight_layout()`으로 간격 자동 조정. Figure: 전체 캔버스 / Axes: 개별 그래프 영역
- 머신러닝 학습 모니터링(손실 추이, 정확도 변화, 학습률 스케줄 동시 확인)에 활용

---

#### 3-9. 벤 다이어그램 (Venn Diagram)

> 여러 집합 간의 관계(교집합, 합집합, 차집합)를 원의 겹침으로 표현해 "서로 다른 집합 간 공통 요소와 차이를 직관적으로 파악"하는 시각화

**핵심**
- 교집합 영역이 클수록 공통 요소가 많음. `matplotlib_venn`의 `venn2`, `venn3` 사용
- AI에서는 훈련/검증 데이터 간 중복 여부 확인(데이터 누수 방지)에 활용

---

### Day 4 (05/28) — 멘토링

> 이날은 개념 학습보다 멘토링 위주였다. 진로·학습법에 대한 조언을 정리한다.

- **Runpod (GPU Cloud)**: 3기 때 사용 경험 — 팀프로젝트 비용에 Runpod 비용을 AWS에 추가하는 형태가 될 가능성. LLM API를 쓰는 형태가 될 수도 있음.
- **추천시스템**: 기계학습을 안 써도 추천은 잘 만들 수 있다. 상당히 굵직하고 각광받는 연구 분야. 지금 코스에서 ML/DL/AI 스킬을 쌓은 뒤, 추천을 할 수 있는 회사에서 경험을 쌓는 걸 추천받음. 잠재성을 보는 회사라면 지금도 가능.
- **OS**: 백엔드 포함 분야이니 관심 가져야 할 부분 — 프로세스·스레드·메모리 관리. Unix OS를 배우면 Unix 프로그램의 단위(프로세스 구성·시행, 프로세스가 쪼개진 게 스레드)와 메모리 관리 방식을 알게 됨. 이게 시작점.
- **평가 지표**: 매일 강사와 자신을 평가하고 이유를 쓴다.
  - 정량 평가: 기준이 핵심 — 시험(BM(T), 벤치마크 테스트)을 봐서 맞/틀 검증. 스스로 AI 시험을 위한 시험지를 만들어야 함.
  - 정성 평가: 소감 기반, 맞냐 틀리냐를 사람이 검증하는 경우가 있고 이걸로 모델을 강화하는 방식도 있음. 사람이 검증하면 힘듦 — "검증을 누가 하나?"라는 질문이 남음.
- **코딩 연습법**: 교재 설명 코드 → 미니퀘스트 코드 → 위클리 챌린지 → 아이디어를 내서 미니 프로젝트 파보기 → 아이디어가 없으면 남의 아이디어를 클론해보기(예: SNS 클론). 학습적 측면은 공식 튜토리얼(파이썬, FastAPI)을 따라 쳐보기. 코테는 이론이 충분하면 예제를 따라칠 필요 없이 문제풀이로, 알고리즘·자료구조는 책/사이트를 먼저 보는 게 낫다.
- **디테일 학습**: 다음 주는 line by line으로 진행될 가능성이 높음 — 이런 경험이 줄어들 것.
- **마인드셋**: OS를 알아야 웹 프로그래밍이 내 컴퓨터에서 어떻게 돌아가는지 알 수 있다. 웹이 아니더라도 동작 원리에 계속 관심을 가질 것. 퀀텀 점프는 없다, 한 단계씩!

---

### Day 5 (05/29) — Seaborn + Scipy

#### 5-1. Seaborn

> 통계적 데이터 시각화를 쉽게 구현할 수 있도록 하는 파이썬 라이브러리

**사용 이유**
변수 간 관계, 데이터 분포, 통계적 분석을 하기 위함

**핵심**
- Matplotlib 기반. Pandas와 높은 호환성, 통계적 시각화 기능 제공

```python
import seaborn as sns
import matplotlib.pyplot as plt
```

---

#### 5-2. Categorical Data

> 정해진 그룹이나 레이블을 가지는 데이터

**왜 쓰는가**
그룹 간 차이·패턴을 직관적으로 비교 분석하고자. 연속형 데이터보다 명확하게 구분 가능

**핵심**
- 그룹 간 차이 비교, 데이터 분포 확인, 이상치 탐지, 패턴/트렌드 분석, 데이터 기반 의사결정 지원

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

data = pd.DataFrame({
    "Category": ["A", "A", "B", "B", "C", "C", "C", "A", "B", "C"],
    "Value": [10, 15, 7, 12, 22, 18, 25, 11, 9, 30]
})

sns.barplot(x="Category", y="Value", data=data)   # 막대그래프
plt.show()
sns.boxplot(x="Category", y="Value", data=data)   # 박스 플롯
plt.show()
sns.violinplot(x="Category", y="Value", data=data)  # 바이올린 플롯 (밀도+분포)
plt.show()
sns.stripplot(x="Category", y="Value", data=data, jitter=True)  # 스트립 플롯 (개별 포인트)
plt.show()
```

---

#### 5-3. Continuous Data

> 특정 구간 내에서 이론적으로 무한한 값을 가질 수 있는, 연속적으로 이어지는 데이터

**왜 쓰는가**
수치 관계를 정량적으로 분석하고 변화를 예측하기 위해

**핵심**
- 측정 단위에 따라 세분화 가능, 연속적 변화·경향 분석 가능
- **빈도(Frequency)**: 구간(bin)으로 나눠 데이터 개수 계산 → 히스토그램(`histplot`)
- **확률 밀도(KDE)**: 연속적인 확률 밀도 곡선 → `kdeplot` / `histplot(kde=True)`
- 막대는 빈도, 곡선은 확률 밀도를 나타냄

```python
import seaborn as sns, numpy as np, matplotlib.pyplot as plt
np.random.seed(42)
data = np.random.randn(1000)
sns.histplot(data, bins=30, kde=True, color='darkorange')
plt.show()

x = np.random.rand(100) * 10
y = x + np.random.randn(100)
sns.regplot(x=x, y=y, color='green')  # 산점도 + 회귀선
plt.show()
```

---

#### 5-4. Relational Data

> 두 개 이상의 변수 간 상관관계/패턴을 분석하는 데이터

**왜 쓰는가**
변수 간 관계를 파악해 데이터 기반 예측·최적화·의사결정을 하기 위해

**핵심**
- 수치형 데이터 활용 패턴 찾기, 다변량 분석

```python
import seaborn as sns
tips = sns.load_dataset("tips")
sns.scatterplot(x="total_bill", y="tip", data=tips, color="blue")
sns.pairplot(tips, vars=["total_bill", "tip", "size"], hue="sex", palette="coolwarm")
```

---

#### 5-5. Time Series Data / Resampling / Moving Average

> Time Series: 시간 흐름에 따라 일정 간격으로 측정된 연속적 데이터. Resampling: 시계열 데이터의 시간 간격을 재조정해 다운/업샘플링하는 과정. Moving Average: 일정 구간 평균값을 계산해 변동성을 완화하고 장기 추세를 파악하는 기법.

**핵심**
- 시간을 독립 변수(X축), 관측값을 종속 변수(Y축)로 설정해 분석
- **Resampling**: 다운샘플링(더 긴 간격으로 집계, 평균·합계·최댓값 등) / 업샘플링(더 짧은 간격으로 확장, 보간 필요)
- **Moving Average**: 단순 이동평균(동일 가중치, 장기 흐름 분석) / 지수 이동평균(최근 데이터에 높은 가중치, 금융 시장) / 가중 이동평균(선형적으로 최근 가중치 증가)

```python
df["SMA"] = df["value"].rolling(window=윈도우_크기).mean()
df["EMA"] = df["value"].ewm(span=윈도우_크기, adjust=False).mean()
```

금융 데이터는 `.read_csv()`, `.read_sql()`, `.read_json()`으로 불러온 뒤 `.resample()`, `.rolling()`, `.ewm()`으로 분석 — 시장 움직임 분석·예측에 활용.

---

#### 5-6. Scipy와 정규 분포 (스텁 — 배움일지 원문이 대부분 미기재)

> Scipy: 계산과 통계 분석을 위한 라이브러리. 정규 분포(Normal Distribution): 데이터가 평균을 중심으로 좌우 대칭을 이루며 종형 곡선을 따르는 확률 분포.

**핵심 (원문에 남아있던 메모)**
- 확률 분포: 확률 변수가 취할 수 있는 모든 값과 그 값이 발생할 확률을 나타내는 함수. 모든 가능한 값들의 확률 합은 1
- 정규 분포의 확률 밀도 함수: 연속형 확률 분포에서 특정 구간에 값이 나타날 가능성을 설명. pdf 값이 크면 해당 구간에 데이터가 몰려있을 확률이 높다는 뜻이며, 전체 면적은 항상 1

> ⚠️ 이 날 Scipy·정규분포 항목은 배움일지 원문에서도 "한 줄 정의"·"핵심"·"코드" 칸이 대부분 비어 있었다(당일 미완성 상태). 위 내용이 원문에서 확인 가능한 전부다.

---

## 🔗 이번 주 개념 흐름

```
[화 05/26] Numpy(배열/브로드캐스팅) + Pandas(Series/DataFrame, Filtering/Grouping)
    ↓ (숫자를 다뤘으니 이제 눈으로 보자)
[수 05/27] 데이터 시각화 개론 — 정형/비정형 구분 → Matplotlib → 막대/히스토그램/산점도/박스플롯 → 다중그래프/벤다이어그램
    ↓ (기술 학습 중간, 방향 점검)
[목 05/28] 멘토링 — GPU 클라우드, 추천시스템, OS, 평가지표, 코딩 연습법
    ↓ (다시 시각화로, 이번엔 통계 지향 도구로)
[금 05/29] Seaborn(범주형/연속형/관계형/시계열) + Scipy(정규분포, 스텁)
```

이번 주는 "데이터를 숫자 배열로 다루는 법(Numpy)"에서 시작해 "그 데이터를 어떻게 눈으로 보여줄 것인가(Matplotlib → Seaborn)"로 확장된 한 주였다. 화요일 Numpy/Pandas가 배열·표 형태 데이터를 다루는 기초 체력이었다면, 수요일은 그걸 그래프로 바꾸는 첫 도구(Matplotlib)를, 금요일은 통계적 시각화에 특화된 두 번째 도구(Seaborn)를 배웠다. Matplotlib이 "직접 하나하나 그리는 도구"였다면 Seaborn은 Pandas DataFrame과 통계 개념(범주형/연속형/관계형/시계열)을 더 자연스럽게 반영하는 상위 도구라는 점에서 진화 관계로 볼 수 있다. 목요일 멘토링은 기술 학습과는 결이 다르지만, "평가 지표를 정량/정성으로 나눠 스스로 검증하라"는 조언이 이번 주 내내 배운 "데이터를 보고 판단하는 법"과 같은 맥락으로 읽힌다.

---

