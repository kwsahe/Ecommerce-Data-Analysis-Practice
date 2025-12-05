# 🛍️ E-commerce Customer Repurchase Prediction
> **이커머스 고객 재구매 예측 및 타겟 마케팅 최적화 프로젝트**

## 1. 프로젝트 개요 (Project Overview)
이 프로젝트는 이커머스 트랜잭션(로그) 데이터를 분석하여 **다음 달에 구매할 확률이 높은 고객을 예측**하는 머신러닝 모델링 프로젝트입니다.
단순한 예측을 넘어, **RFM 분석**을 통해 고객의 행동 패턴을 정량화하고, 마케팅 비용 대비 수익(ROI)을 극대화할 수 있는 **타겟 고객군(Target Audience)**을 도출하는 것을 목표로 했습니다.

* **진행 기간:** 2025.12.xx ~ 2025.12.xx
* **사용 언어/툴:** Python, Pandas, Scikit-learn, Matplotlib, Seaborn
* **핵심 기법:** RFM Analysis, Feature Engineering, GridSearch Tuning, ROI Calculation

## 2. 문제 정의 (Business Problem)
* **현상:** 무작위적인 쿠폰 배포(Random Targeting)로 인한 마케팅 비용 낭비 발생.
* **데이터:** 고객의 과거 행동 로그(Transaction Level)만 존재하며, 유저 단위(User Level)의 인사이트가 부재함.
* **목표:** 머신러닝 모델을 통해 구매 확률이 높은 고객을 선별하여 **마케팅 효율(Profit)**을 높이고자 함.

## 3. 분석 및 해결 과정 (Process)

### Step 1. 데이터 전처리 & 피처 엔지니어링
* **데이터 관점 전환:** `Transaction` 단위의 로그 데이터를 `User` 단위로 요약(Aggregation).
* **RFM 분석:** 고객 가치 평가를 위한 핵심 지표 생성.
    * `Recency`: 마지막 구매일로부터 경과 기간
    * `Frequency`: 구매 빈도
    * `Monetary`: 총 구매 금액
* **파생 변수 생성:**
    * `Tenure`: 가입 기간 (첫 구매일로부터 경과일)
    * `AOV`: 객단가 (평균 주문 금액)
    * `AgeGroup`: 연령대 구간화 (Binning) 및 원-핫 인코딩

### Step 2. 모델링 및 하이퍼파라미터 튜닝
* **알고리즘:** RandomForestClassifier (해석 용이성 및 안정성 고려)
* **최적화 (GridSearchCV):** 과적합(Overfitting) 방지를 위한 교차 검증(CV) 및 튜닝 수행.
    * *Best Params:* `max_depth: 3`, `n_estimators: 200`
    * *해석:* 복잡한 트리 구조보다 얕은 깊이에서 최적 성능을 보임 → 고객의 구매 패턴이 `Recency` 등 특정 핵심 변수에 의해 명확히 구분됨을 시사.

### Step 3. 비즈니스 임팩트 분석 (ROI Calculation)
* 단순 정확도(Accuracy)가 아닌 **비즈니스 수익(Profit)** 관점에서 모델 평가.
* **Confusion Matrix 기반 시뮬레이션:**
    * `쿠폰 비용`: 1,000원 / `판매 이익`: 10,000원 가정
    * **결과:** 랜덤 타겟팅 대비 AI 모델 도입 시 마케팅 비용 절감 및 순수익 증대 효과 확인.

## 4. 분석 결과 (Key Insights)
1.  **Feature Importance:** 분석 결과, 재구매를 결정짓는 가장 중요한 요인은 **'최근성(Recency)'**으로 나타남. (최근에 방문한 고객일수록 재구매 확률 급증)
2.  **Action Plan:** 마케팅 예산을 신규 유입보다 **최근 30일 이내 방문 이력이 있는 고객의 리텐션**에 집중하는 것이 효율적임.

## 5. 프로젝트 구조 (Directory Structure)
```text
📂 Ecommerce-Data-Analysis
 ┣ 📂 01_EDA_Visualization    # 데이터 탐색 및 시각화
 ┣ 📂 02_Feature_Engineering  # RFM 변환 및 파생변수 생성 코드 (rfm_analysis.ipynb)
 ┣ 📂 03_Modeling             # 모델 학습 및 튜닝
 ┣ 📂 data                    # 원본 및 가공 데이터
 ┗ 📜 README.md               # 프로젝트 결과 요약 보고서