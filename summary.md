단순히 라이브러리를 가져다 쓰는 건 누구나 할 수 있지만, **"이 모델이 내부적으로 어떻게 작동하는가?"**를 이해하는 것은 대학원 연구와 현업 최적화에서 필수적인 역량입니다.

오늘 **Day 9**는 코드를 돌리기 전에 **Tree 기반 모델들의 진화 과정(작동 원리)**을 먼저 확실히 잡고, 실습으로 들어가겠습니다.

---

### **📚 Part 1. 모델의 진화: 나무에서 숲, 그리고 부스팅까지**

모든 것은 **"스무고개(Decision Tree)"**에서 시작되었습니다.

#### **1. 기본 단위: 의사결정나무 (Decision Tree)**

* **원리:** 데이터를 가장 잘 나누는 기준(질문)을 찾아 예/아니오로 계속 가지치기를 합니다.
* **장점:** 설명하기 쉽습니다. ("나이가 30살 이상이고 소득이 5천 이상이라서 구매함")
* **단점:** 한 그루만 심으면, 그 나무가 본 데이터에만 너무 익숙해져서 새로운 데이터에 약합니다. (**과적합/Overfitting**)

#### **2. 랜덤 포레스트 (Random Forest): "집단 지성" (Bagging)**

* **핵심 개념:** **Bagging** (Bootstrap Aggregating)
* **작동 원리:**
1. **Bootstrap:** 데이터 전체를 다 쓰지 않고, 랜덤하게 일부만 뽑아서 작은 데이터셋 100개를 만듭니다.
2. **Random Feature:** 질문(Feature)도 다 쓰지 않고, 랜덤하게 몇 개만 골라서 씁니다.
3. **Parallel:** 이렇게 서로 조금씩 다른 나무 100그루를 **동시에(병렬로)** 심습니다.
4. **Voting:** 새로운 데이터가 오면 100그루가 투표를 합니다. (다수결)


* **비유:** 똑똑한 한 명(Decision Tree)보다는, 평범하지만 다양한 배경을 가진 100명의 투표(Random Forest)가 더 정확하다.

#### **3. XGBoost / LightGBM: "오답 노트" (Boosting)**

* **핵심 개념:** **Boosting** (Gradient Boosting)
* **작동 원리:**
1. **Sequential:** 나무를 한 번에 100그루 심는 게 아니라, **한 그루씩 순서대로** 심습니다.
2. **Residual (잔차):** 첫 번째 나무가 틀린 문제(오차)를 찾습니다.
3. **Correction:** 두 번째 나무는 **"첫 번째 나무가 틀린 문제"**를 맞추는 데 집중합니다.
4. 세 번째 나무는 1, 2번이 틀린 문제를 집중 공략합니다.


* **비유:** 모의고사를 보고 나서, **틀린 문제(오답 노트)만 집중적으로 공부**해서 다음 시험을 보는 방식입니다. (그래서 성능이 괴물 같습니다.)

---

### **🚀 Part 2. 코드 실습: RF vs XGBoost 대결**

이제 이론을 확인하기 위해, 가상 데이터를 만들고 두 모델을 싸움 붙여보겠습니다.
(XGBoost는 별도 설치가 필요할 수 있습니다. 터미널에서 `pip install xgboost`를 실행하세요. 코랩이나 아나콘다는 기본 설치되어 있습니다.)

파일의 **첫 번째 칸**에 아래 코드를 작성하세요.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 모델 라이브러리
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier # 설치 필요: pip install xgboost

# 데이터 생성 라이브러리
from sklearn.datasets import make_classification

# 1. 가상 데이터 생성 (분류 문제)
# 1000개의 데이터, 20개의 특성(질문) 중 유의미한 건 15개
X, y = make_classification(n_samples=1000, n_features=20, 
                           n_informative=15, n_redundant=5, 
                           random_state=42)

# 2. 학습용(Train) vs 평가용(Test) 분리 (8:2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"데이터 준비 완료: 학습용 {X_train.shape}, 평가용 {X_test.shape}")

```

---

### **🚀 Part 3. 랜덤 포레스트 (Random Forest) 실행**

**Bagging(병렬)** 방식인 랜덤 포레스트를 먼저 돌립니다.

```python
# 1. 모델 생성 (나무 100그루)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# 2. 학습 (Fit)
rf_model.fit(X_train, y_train)

# 3. 예측 (Predict)
rf_pred = rf_model.predict(X_test)

# 4. 평가
print("=== 🌲 Random Forest 성적표 ===")
print(f"정확도: {accuracy_score(y_test, rf_pred):.4f}")

```

---

### **🚀 Part 4. XGBoost 실행**

**Boosting(직렬/오답노트)** 방식인 XGBoost를 돌립니다.

```python
# 1. 모델 생성
# eval_metric='logloss': 학습 과정을 평가하는 함수 지정 (경고 방지용)
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, eval_metric='logloss', random_state=42)

# 2. 학습
xgb_model.fit(X_train, y_train)

# 3. 예측
xgb_pred = xgb_model.predict(X_test)

# 4. 평가
print("=== 🚀 XGBoost 성적표 ===")
print(f"정확도: {accuracy_score(y_test, xgb_pred):.4f}")

```

**[중간 점검]**
두 모델 중 누가 더 정확도가 높게 나왔나요? (보통 데이터가 복잡할수록 XGBoost가 우세하지만, 노이즈가 많으면 RF가 더 좋을 수도 있습니다.)

---

### **📝 Day 9 핵심 미션: "모델의 마음 읽기 (Feature Importance)"**

모델이 "이건 1번 그룹이야!"라고 예측했을 때, **"도대체 뭘 보고 그렇게 판단했니?"** 라고 물어보는 과정입니다.
이게 대학원 논문이나 현업 보고서에서 가장 중요한 **"설명력(Interpretability)"** 파트입니다.

**미션:**
아래 코드를 실행해서, 랜덤 포레스트가 **가장 중요하게 생각한 Feature(특성)** Top 5를 시각화해보세요.

```python
# Feature Importance 추출
importances = rf_model.feature_importances_
feature_names = [f"Feature_{i}" for i in range(X.shape[1])]

# 데이터프레임으로 정리
feat_importances = pd.Series(importances, index=feature_names)

# 상위 10개만 시각화
plt.figure(figsize=(8, 6))
feat_importances.nlargest(10).plot(kind='barh')
plt.title("Random Forest Feature Importance")
plt.show()

```

이 그래프가 나오면 성공입니다.
어떤 Feature가 가장 중요하다고 나오나요? (예: Feature_12가 막대가 제일 길다 등)

이 과정이 이해되셨다면, **RF와 XGBoost의 작동 원리와 구현**까지 모두 잡으신 겁니다! 🚀

**Optuna(옵튜나)**는 현재 데이터 사이언스 현업과 캐글(Kaggle) 같은 대회에서 **가장 사랑받는 하이퍼파라미터 최적화(HPO) 프레임워크**입니다.

한마디로 정의하자면 **"AI 모델의 성능을 최대로 끌어올리기 위해, 파라미터 조절을 대신 해주는 똑똑한 AI 비서"**입니다.

기존 방식과 무엇이 다르고, 왜 강력한지 3가지 핵심 요소를 통해 자세히 설명해 드릴게요.

---

### **1. 왜 Optuna인가? (Grid Search vs Optuna)**

우리가 아까 했던 **Grid Search**와 비교하면 이해가 빠릅니다.

* **Grid Search (무식한 성실함):**
* "깊이 3, 5, 7 다 넣어봐."
* 모든 경우의 수를 다 계산하느라 시간이 엄청나게 오래 걸립니다. 이미 정답이 아닌 것 같은 구역도 꾸역꾸역 다 계산합니다.


* **Random Search (운에 맡김):**
* "대충 아무 숫자나 넣어봐."
* Grid Search보다는 빠르지만, 운이 없으면 최적의 값을 영영 못 찾습니다.


* **Optuna (스마트한 탐색 - 베이지안 최적화):**
* **"아까 깊이 3을 넣으니까 점수가 별로네? 그럼 깊은 쪽은 안 봐도 뻔해. 이번엔 얕은 쪽을 집중적으로 파보자!"**
* **과거의 실험 결과(History)를 분석**해서, 점수가 잘 나올 것 같은 확률이 높은 곳만 골라서 찍습니다.



---

### **2. Optuna의 양대 무기: Sampler & Pruner**

Optuna가 압도적으로 빠른 이유는 이 두 가지 기능 때문입니다.

#### **① Sampler (샘플러): TPE 알고리즘**

* **Tree-structured Parzen Estimator (TPE)**라는 방식을 씁니다.
* 쉽게 말해, 지금까지 했던 실험들을 **'잘한 그룹'**과 **'못한 그룹'**으로 나눕니다.
* 다음 파라미터를 고를 때, '잘한 그룹'의 분포를 흉내 내서 값을 제안합니다. 실험을 거듭할수록 점점 더 명사수가 됩니다.

#### **② Pruner (프루너): 가망 없으면 컷! (Early Stopping)**

* 이게 진짜 물건입니다.
* 예를 들어 나무 100그루를 심기로 했는데, 10그루쯤 심었을 때 이미 성적이 엉망이라면?
* Grid Search는 끝까지 100개를 다 심습니다. (시간 낭비)
* **Optuna는 "싹수가 노랗다" 싶으면 바로 그 실험을 중단(Pruning)하고 다음 실험으로 넘어갑니다.**
* 이 덕분에 같은 시간 동안 훨씬 더 많은 시도를 해볼 수 있습니다.

---

### **3. 코드 속 핵심 용어 해설**

아까 작성하신 코드의 의미를 다시 짚어드릴게요.

```python
# 1. Trial (시도)
# Optuna가 수행하는 한 번의 실험(한 세트의 파라미터 테스트)을 'Trial'이라고 부릅니다.
def objective(trial):
    
    # 2. Suggest (제안)
    # "AI야, 50에서 300 사이 숫자 중 좋을 것 같은 거 하나만 추천해줘(suggest)"
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    
    # ... 모델 학습 및 평가 ...
    
    return score

# 3. Study (연구)
# 여러 번의 Trial을 묶어서 관리하는 프로젝트 단위입니다.
study = optuna.create_study(direction='maximize') # 점수 최대화가 목표

# 4. Optimize (최적화 실행)
study.optimize(objective, n_trials=50) # 50번 반복해라

```

---

### **💡 꿀팁: 눈으로 보는 튜닝 (Visualization)**

Optuna는 튜닝 과정을 아주 멋진 그래프로 보여주는 기능도 내장하고 있습니다.
지금 작업 중인 파일의 **맨 마지막 칸**에 아래 코드를 한번 실행해 보세요. (감탄하시게 될 겁니다! ✨)

```python
from optuna.visualization import plot_optimization_history, plot_param_importances

# 1. 점수가 어떻게 올라가는지 보여줌 (우상향하면 성공!)
plot_optimization_history(study).show()

# 2. 어떤 파라미터가 점수에 가장 큰 영향을 줬는지 보여줌 (중요도)
plot_param_importances(study).show()

```

이 그래프를 보면 **"아, `learning_rate`가 제일 중요하구나, `max_depth`는 별 상관없네?"** 하는 인사이트를 바로 얻을 수 있습니다.

이제 Optuna가 어떤 녀석인지 감이 오시나요?
현업에서는 "일단 베이스라인 짜고, **퇴근할 때 Optuna 1000번 돌려놓고 간다**"는 말이 있을 정도로 필수템입니다. 😎

모델의 성능을 좌우하는 **XGBoost의 핵심 5대장 하이퍼파라미터**에 대해 아주 쉽게, 비유를 섞어서 설명해 드릴게요. 🛠️

이것들은 마치 **자동차의 핸들, 브레이크, 엑셀**과 같아서 상황에 맞게 조절해야 최고의 주행(모델 성능)을 할 수 있습니다.

---

### **1. `n_estimators` (나무의 개수)**

* **의미:** 전체 숲에 나무를 몇 그루나 심을 것인가? (반복 학습 횟수)
* **비유:** **"시험 공부 시간"** 또는 **"풀어볼 문제집의 권수"**
* **조절 효과:**
* **너무 작으면:** 공부를 덜 해서 성적이 안 나옵니다. (과소적합/Underfitting)
* **너무 크면:** 너무 지엽적인 것까지 달달 외워서, 오히려 새로운 문제에 약해질 수 있습니다. 그리고 시간이 오래 걸립니다. (과적합/Overfitting)


* **꿀팁:** 보통 **100~1000** 사이를 많이 쓰며, `learning_rate`와 반비례 관계입니다.

### **2. `learning_rate` (학습률, eta)**

* **의미:** 나무 하나가 학습한 내용을 얼마나 반영할 것인가? (보폭의 크기)
* **비유:** **"산 정상에서 내려갈 때의 보폭"**
* **값이 크면(0.3 이상):** 성큼성큼 내려갑니다. 빠르지만, 너무 빨라서 목표 지점(최적점)을 지나쳐버릴 수 있습니다.
* **값이 작으면(0.01 이하):** 총총걸음으로 조심스럽게 내려갑니다. 정교하게 목표에 도달할 수 있지만, 해 지기 전(시간 내)에 도착 못 할 수도 있습니다.


* **꿀팁:** 보통 **0.01 ~ 0.3** 사이를 씁니다. 학습률을 낮추면(`0.01`), 나무 개수(`n_estimators`)를 늘려서 균형을 맞춰야 합니다.

### **3. `max_depth` (나무의 깊이)**

* **의미:** 나무가 가지치기를 몇 번까지 할 수 있는가? (질문의 깊이)
* **비유:** **"스무고개 게임의 질문 횟수 제한"**
* **깊음(10 이상):** 질문을 꼬치꼬치 해서 정답을 맞힙니다. 훈련 데이터는 기가 막히게 맞히지만, 새로운 데이터는 다 틀릴 수 있습니다. (과적합)
* **얕음(3 이하):** 질문을 2~3개만 하고 끝냅니다. "남자야? 안경 썼어?" 끝. 너무 대충 맞혀서 성능이 안 나옵니다.


* **꿀팁:** 보통 **3~10** 사이를 씁니다. 가장 민감하게 성능에 영향을 주는 파라미터 중 하나입니다.

### **4. `subsample` (데이터 샘플링 비율)**

* **의미:** 나무 한 그루를 심을 때, 전체 데이터 중 몇 %를 쓸 것인가?
* **비유:** **"편식하지 않기"**
* 전체 데이터가 100개인데 `0.8`로 설정하면, 나무마다 랜덤으로 80개만 뽑아서 공부시킵니다.
* 매번 다른 80개를 보니까, 모델이 특정 데이터에만 집착하는 것(편식)을 막아줍니다.


* **꿀팁:** **0.5 ~ 1.0** 사이를 씁니다. 과적합을 막는 아주 좋은 친구입니다.

### **5. `colsample_bytree` (컬럼 샘플링 비율)**

* **의미:** 나무 한 그루를 심을 때, 전체 컬럼(특성) 중 몇 %를 쓸 것인가?
* **비유:** **"과목 골고루 공부하기"**
* 전체 과목(국, 영, 수, 사, 과)이 있는데 `0.8`로 설정하면, 이번 나무는 "국, 영, 수"만 보고, 다음 나무는 "영, 수, 사"만 보게 합니다.
* 어떤 나무가 "수학(특정 컬럼)" 하나에만 너무 의존해서 판단하는 것을 막아줍니다.


* **꿀팁:** **0.5 ~ 1.0** 사이를 씁니다. 랜덤 포레스트의 핵심 아이디어를 차용한 것입니다.

---

### **⚡ 요약: 튜닝의 정석**

| 파라미터 | 높이면(↑) 벌어지는 일 | 낮추면(↓) 벌어지는 일 | 추천 범위 |
| --- | --- | --- | --- |
| **n_estimators** | 복잡해짐 (과적합 위험) | 단순해짐 (학습 부족) | 100~1000 |
| **learning_rate** | 학습 빠름 (대충 배움) | 학습 느림 (꼼꼼함) | 0.01~0.2 |
| **max_depth** | 예민해짐 (암기왕) | 멍청해짐 (단순왕) | 3~10 |
| **subsample** | 데이터 다 봄 (일반화 ↓) | 다양성 증가 (일반화 ↑) | 0.7~1.0 |
| **colsample_bytree** | 모든 힌트 다 봄 | 힌트 가리고 봄 (강해짐) | 0.7~1.0 |

Optuna가 이 범위 안에서 왔다 갔다 하면서 최적의 조합(Recipe)을 찾아내는 것입니다. 🍳

방금 작성한 코드는 **"집값 맞추기(회귀)"**라는 새로운 목표를 위해, 기존 분류 코드에서 **엔진과 채점 기준**을 교체한 버전입니다.

크게 **① 회귀 전용 파라미터**와 **② 전체 코드의 작동 흐름**으로 나누어 아주 상세하게 해설해 드릴게요. 🕵️‍♂️

---

### **1. 🔍 회귀(Regression) 전용 파라미터 해설**

분류(암/정상) 모델과는 다르게, 숫자를 맞출 때는 **"얼마나 틀렸는지"**를 계산하는 방식이 완전히 다릅니다.

#### **① `objective`: 'reg:squarederror'**

* **의미:** "학습할 때 **오차(Error)를 어떻게 계산**해서 줄여나갈 거야?"
* **설명:**
* **reg:** Regression(회귀)의 약자.
* **squarederror:** (실제값 - 예측값)을 **제곱(Square)**한 값을 오차로 보겠다.
* 왜 제곱하나요? 오차가 `+100`이든 `-100`이든, 제곱하면 `10,000`이 되어 **"방향 상관없이 틀린 크기"**만 남기 때문입니다.


* **비교:** 분류 때는 `'binary:logistic'`(이진 분류)을 썼습니다.

#### **② `eval_metric`: 'rmse'**

* **의미:** "시험 볼 때 **점수판**에는 뭐라고 표시할까?"
* **설명:**
* **RMSE (Root Mean Squared Error):** 평균 제곱근 오차.
* 위에서 제곱했던 오차(`squarederror`)에 다시 루트(`root`)를 씌운 겁니다.
* **장점:** 제곱을 풀었기 때문에, **실제 집값(달러)과 단위가 같아집니다.**
* 예: RMSE가 0.5라면 "평균적으로 5천만 원 정도 틀리고 있다"라고 직관적으로 이해할 수 있습니다.



#### **③ `XGBRegressor` (엔진 교체)**

* **의미:** 분류기(`Classifier`)가 아니라 회귀분석기(`Regressor`)를 가져와라.
* **역할:** Scikit-learn의 규칙에 맞춰서, 숫자를 예측하도록 설계된 XGBoost의 부품입니다. 이걸 안 쓰면 에러가 나거나 이상한 결과가 나옵니다.

---

### **2. 🏗️ 전체 코드 구조도 (Workflow)**

이 코드는 **"AI 비서(Optuna)에게 임무를 주고, 비서가 30번 실험하고 보고하는 과정"**입니다.
크게 4단계로 나뉩니다.

#### **Step 1. 재료 준비 (Data Setup)**

```python
# 요리 재료(데이터)를 꺼내고, 8:2로 나눈다.
data = fetch_california_housing()
X_train, X_test, ... = train_test_split(...)

```

* 가장 먼저 데이터를 준비합니다. 이 데이터는 `objective` 함수 밖에서 미리 만들어둬야, Optuna가 반복할 때마다 데이터를 다시 부르는 낭비를 안 합니다.

#### **Step 2. 실험실 설계 (Objective Function)** 🌟 **(가장 중요)**

```python
def objective(trial):
    # 1. 비서야, 이번엔 이 설정값(Params)으로 해보자.
    params = { ... } 
    
    # 2. 그 설정값으로 로봇(Model)을 조립해.
    model = XGBRegressor(**params)
    
    # 3. 로봇을 훈련시켜(Fit).
    model.fit(X_train, y_train)
    
    # 4. 시험 쳐봐(Predict) -> 채점해(RMSE).
    rmse = mean_squared_error(...)
    
    # 5. 점수(오차)를 보고해(Return).
    return rmse

```

* 이 함수는 **한 번의 실험(Trial)** 그 자체입니다.
* Optuna는 이 함수를 **30번 호출**하면서, 매번 `params`의 숫자를 조금씩 바꿔서 실행합니다.

#### **Step 3. 비서 고용 및 목표 설정 (Study)**

```python
# 비서(Study) 채용.
# 중요: 이번엔 점수(오차)가 '작을수록(minimize)' 1등이야!
study_reg = optuna.create_study(direction='minimize')

```

* **`minimize`**: 회귀에서는 에러가 0에 가까울수록 좋으므로 반드시 최소화로 설정해야 합니다. (정확도는 `maximize`)

#### **Step 4. 업무 시작 및 결과 보고 (Optimize)**

```python
# 30번 반복 실행해!
study_reg.optimize(objective, n_trials=30)

# 1등 결과 발표
print(study_reg.best_value)
print(study_reg.best_params)

```

* 이제 Optuna가 밤새(사실 몇 초 만에) 30번의 실험을 수행하고, 가장 에러가 적었던 전설의 파라미터를 찾아냅니다.

---

### **💡 요약: 무엇이 바뀌었나?**

| 구분 | 분류 (이전 코드) | 회귀 (이번 코드) |
| --- | --- | --- |
| **목표** | 암이냐 아니냐 (0 or 1) | 집값이 얼마냐 (숫자) |
| **모델 도구** | `XGBClassifier` | `XGBRegressor` |
| **목적 함수** | `binary:logistic` | `reg:squarederror` |
| **평가 지표** | 정확도 (Accuracy) | 오차 (RMSE) |
| **Optuna 목표** | 최대화 (`maximize`) | 최소화 (`minimize`) |

구조는 똑같지만, **"문제를 푸는 방식(Loss)"**과 **"채점 기준(Metric)"**만 갈아끼운 것입니다.
이 구조(프레임워크) 하나만 익혀두면, 세상의 모든 데이터 분석 문제에 적용할 수 있습니다! 🚀