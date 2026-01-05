import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import os
import warnings
import platform

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title = "서울 부동산 예측",
    page_icon = "🏠",
    layout = "wide"
)

if platform.system() == 'Windows':
    plt.rc('font', family='D2Coding')
else:
    plt.rc('font', family='D2Coding')
plt.rcParams['axes.unicode_minus'] = False

current_dir = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(current_dir, '..', 'data')

st.write(f"데이터 참조 경로: {BASE_PATH}")

FILE_LIST = {
    'Sale_Change': ("'25.12월 4주 주간아파트가격동향조사 시계열_매매변동률.csv", False),
    'Jeonse_Change': ("'25.12월 4주 주간아파트가격동향조사 시계열_전세변동률.csv", False),
}

def parse_custom_date(date_series):
    dates = []
    current_year = 2012
    last_month = 5
    for item in date_series:
        if pd.isna(item):
            dates.append(pd.NaT)
            continue
        item = str(item).strip().replace("'", "")
        if item.count('.') == 2:
            try:
                parts = item.split('.')
                y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
                current_year = 2000 + y
                last_month = m
                dates.append(pd.Timestamp(year=current_year, month=m, day=d))
            except:
                dates.append(pd.NaT)
            continue
        try:
            parts = item.split('.')
            m, d = int(parts[0]), int(parts[1])
            if m < last_month:
                current_year += 1
            last_month = m
            dates.append(pd.Timestamp(year=current_year, month=m, day=d))
        except:
            dates.append(pd.NaT)
    return dates

@st.cache_data
def load_all_data():
    master_df = None
    
    # 파일 경로가 실제로 존재하는지 먼저 체크
    if not os.path.exists(BASE_PATH):
        st.error(f"❌ 데이터 폴더를 찾을 수 없습니다: {BASE_PATH}")
        return pd.DataFrame()

    total_files = len(FILE_LIST)
    
    # 로딩 바 생성
    progress_text = "데이터를 불러오는 중입니다..."
    my_bar = st.progress(0, text=progress_text)
    
    idx = 0
    for col_name, (filename, is_cat) in FILE_LIST.items():
        file_path = os.path.join(BASE_PATH, filename)
        
        if not os.path.exists(file_path):
            st.warning(f"⚠️ 파일 없음: {filename}")
            continue

        try:
            nrows = 4 if is_cat else 3
            try:
                df_header = pd.read_csv(file_path, encoding='cp949', header=None, nrows=nrows)
            except:
                df_header = pd.read_csv(file_path, encoding='utf-8', header=None, nrows=nrows)
            
            if is_cat:
                row_region = df_header.iloc[1]
                row_sub = df_header.iloc[3]
            else:
                row_region = df_header.iloc[1]
                row_sub = df_header.iloc[2]
            
            cols = []
            current_region = None
            for i in range(len(row_region)):
                r1 = row_region[i]
                sub = row_sub[i]
                if pd.notna(r1):
                    current_region = str(r1).replace('\n', '').split('(')[0].strip()
                if pd.notna(sub):
                    clean_sub = str(sub).replace('\n', '').strip()
                    if current_region:
                        col_name_new = current_region if clean_sub == current_region else f"{current_region}_{clean_sub}"
                    else:
                        col_name_new = clean_sub
                else:
                    col_name_new = current_region if current_region else f"Unknown_{i}"
                cols.append(col_name_new)

            try:
                df = pd.read_csv(file_path, encoding='cp949', header=None, skiprows=5)
            except:
                df = pd.read_csv(file_path, encoding='utf-8', header=None, skiprows=5)

            # 길이 맞추기
            min_len = min(len(cols), len(df.columns))
            df = df.iloc[:, :min_len]
            cols = cols[:min_len]
            df.columns = cols
            
            # 날짜 처리
            df.rename(columns={df.columns[0]: 'Date_Raw'}, inplace=True)
            df['Date'] = parse_custom_date(df['Date_Raw'])
            df = df.drop(columns=['Date_Raw'])
            
            # Melt
            df_melted = df.melt(id_vars=['Date'], var_name='Region', value_name=col_name)
            df_melted = df_melted.dropna(subset=['Date'])

            if master_df is None:
                master_df = df_melted
            else:
                master_df = pd.merge(master_df, df_melted, on=['Date', 'Region'], how='outer')
        except Exception as e:
            st.error(f"에러 발생 ({filename}): {e}")

        idx += 1
        my_bar.progress(idx / total_files, text=f"로딩 중... {filename}")

    my_bar.empty()
    
    if master_df is not None:
        return master_df.sort_values(['Region', 'Date']).reset_index(drop=True)
    return pd.DataFrame()

st.title("🏠 AI 부동산 시장 예측 결과 (2026 Ver.)")
st.markdown("""
이 대시보드는 **Prophet 시계열 머신러닝 모델**을 사용하여 서울/경기/지방 아파트의 **매매 및 전세 가격 흐름**을 분석하고 예측합니다.
""")

df = load_all_data()

if df.empty:
    st.error("데이터를 불러오지 못했습니다. data 폴더 위치와 파일명을 확인해주세요.")
else:
    st.sidebar.header("🔍 지역 선택")
    all_regions = sorted(df['Region'].unique().astype(str))

    default_idx = 0
    if '서울' in all_regions:
        default_idx = all_regions.index('서울')

    selected_region = st.sidebar.selectbox("어디가 궁금하신가요?", all_regions, index=default_idx)
    st.sidebar.success(f"선택됨: **{selected_region}**")
    
    st.sidebar.markdown("---")
    st.sidebar.caption("💡 Tip: 그래프 위의 탭을 눌러 '2026년 상세'를 확인하세요.")

    def plot_prophet_chart(data, target_col, title, color):
        # 학습 데이터 준비
        prophet_df = data[['Date', target_col]].dropna().rename(columns={'Date': 'ds', target_col: 'y'})
        
        if len(prophet_df) < 30:
            st.warning("데이터가 부족하여 예측할 수 없습니다.")
            return

        # 모델 학습
        with st.spinner(f"'{title}' 예측 모델 돌아가는 중..."):
            m = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=True)
            m.fit(prophet_df)
            future = m.make_future_dataframe(periods=60, freq='W')
            forecast = m.predict(future)
        
        # 2026년 필터링
        fc_2026 = forecast[forecast['ds'].dt.year == 2026]
        avg_2026 = fc_2026['yhat'].mean()
        
        # KPI 지표 (숫자 크게 보여주기)
        status = "상승장 🔥" if avg_2026 > 0 else "하락/조정장 ❄️"
        st.metric(label=f"2026년 예상 {title}", value=f"{avg_2026:.3f}%", delta=status)
        
        # 탭 구성
        tab1, tab2 = st.tabs(["📊 전체 흐름 (2012~2026)", "🔎 2026년 확대 보기"])
        
        with tab1:
            fig1 = m.plot(forecast)
            plt.title(f"{selected_region} {title} 장기 추세", fontsize=15)
            plt.axhline(0, color='red', linestyle='--')
            plt.xlabel("연도")
            plt.ylabel("변동률 (%)")
            st.pyplot(fig1)
            
        with tab2:
            fig2, ax = plt.subplots(figsize=(10, 5))
            ax.plot(fc_2026['ds'], fc_2026['yhat'], color=color, linewidth=2, label='예측값')
            ax.fill_between(fc_2026['ds'], fc_2026['yhat_lower'], fc_2026['yhat_upper'], color=color, alpha=0.2, label='오차범위')
            ax.axhline(0, color='red', linestyle='--', linewidth=1)
            ax.set_title(f"2026년 {selected_region} {title} 상세 전망", fontsize=15)
            ax.grid(True, alpha=0.3)
            ax.legend()
            st.pyplot(fig2)

    # 선택된 지역 데이터만 추출
    region_data = df[df['Region'] == selected_region].copy()
    
    # 2단 레이아웃 (매매 / 전세)
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("📈 매매 가격 전망")
        plot_prophet_chart(region_data, 'Sale_Change', '매매 변동률', '#FF5733') # 주황/빨강
        
    with col_b:
        st.subheader("📉 전세 가격 전망")
        plot_prophet_chart(region_data, 'Jeonse_Change', '전세 변동률', '#3375FF') # 파랑