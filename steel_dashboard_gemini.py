import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
import warnings
import json
import re
import traceback
import uuid
from datetime import datetime
from google_sheets_full import render_full_history_ui, add_to_full_history

# 경고 메시지 무시
warnings.filterwarnings('ignore')

# --- 페이지 설정 ---
st.set_page_config(page_title="철강 설비 AI 대시보드", layout="wide")
st.title(" 철강 설비 AI 에이전트 대시보드")

# 세션 상태 초기화
if 'error_logs' not in st.session_state:
    st.session_state.error_logs = []
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

# --- 에러 로깅 함수 ---
def log_error(error_type, error_msg, details=None):
    """에러를 세션 상태에 기록"""
    error_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type": error_type,
        "message": str(error_msg),
        "details": details or traceback.format_exc()
    }
    st.session_state.error_logs.append(error_entry)
    return error_entry

# --- 분석 히스토리 추가 함수 ---
def add_to_history(question, result_type, figure=None, data=None, insights=None, code=None, data_code=None):
    """분석 결과를 히스토리에 추가"""
    history_entry = {
        "id": str(uuid.uuid4()),  # 고유 ID 생성
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "result_type": result_type,
        "figure": figure,
        "data": data.to_dict() if data is not None else None,
        "insights": insights,
        "code": code,
        "data_code": data_code
    }
    st.session_state.analysis_history.append(history_entry)

# --- 이상치 제거 함수 ---
def remove_outliers(df, column, method='iqr', threshold=1.5):
    """이상치 제거 함수"""
    original_len = len(df)
    
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        df_clean = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)].copy()
        
        removed = original_len - len(df_clean)
        info = {
            "method": "IQR (사분위수 범위)",
            "Q1": float(Q1),
            "Q3": float(Q3),
            "IQR": float(IQR),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "threshold": threshold
        }
        
    elif method == 'zscore':
        import numpy as np
        mean = df[column].mean()
        std = df[column].std()
        z_scores = np.abs((df[column] - mean) / std)
        
        df_clean = df[z_scores < threshold].copy()
        
        removed = original_len - len(df_clean)
        info = {
            "method": "Z-Score (표준편차)",
            "mean": float(mean),
            "std": float(std),
            "threshold": threshold
        }
        
    elif method == 'percentile':
        lower_percentile = threshold
        upper_percentile = 100 - threshold
        
        lower_bound = df[column].quantile(lower_percentile / 100)
        upper_bound = df[column].quantile(upper_percentile / 100)
        
        df_clean = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)].copy()
        
        removed = original_len - len(df_clean)
        info = {
            "method": f"Percentile (백분위수 {lower_percentile}~{upper_percentile}%)",
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "lower_percentile": lower_percentile,
            "upper_percentile": upper_percentile
        }
    
    else:
        df_clean = df.copy()
        removed = 0
        info = {"method": "제거 안 함"}
    
    return df_clean, removed, info

# --- Gemini LLM 로드 ---
@st.cache_resource
def get_llm():
    """Google Gemini LLM 객체를 로드하는 함수"""
    print("Gemini LLM 로드 시도...")
    
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as e:
        error = log_error("ImportError", "langchain-google-genai 패키지 없음", str(e))
        st.error(f" {error['message']}")
        return None
    
    api_key = st.secrets.get("GOOGLE_API_KEY") if "GOOGLE_API_KEY" in st.secrets else os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        log_error("ConfigError", "Google API 키 없음", "GOOGLE_API_KEY 환경변수 또는 secrets 미설정")
        st.warning("⚠️ Google Gemini API 키가 설정되지 않았습니다. AI 인사이트 없이 기본 분석만 가능합니다.")
        return None
    
    models_to_try = [
        "models/gemini-2.5-flash",
        "models/gemini-2.5-pro-preview-05-06",
        "gemini-2.0-flash-exp",
        "models/gemini-1.5-flash-latest",
        "models/gemini-1.5-pro-latest",
    ]
    
    for model_name in models_to_try:
        try:
            print(f"시도 중: {model_name}")
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0,
                google_api_key=api_key
            )
            
            test = llm.invoke("Hi")
            print(f" 성공: {model_name}")
            st.success(f" Google Gemini ({model_name}) 로드 완료!")
            return llm
            
        except Exception as e:
            log_error("ModelLoadError", f"{model_name} 로드 실패", str(e))
            print(f" 실패: {model_name} - {e}")
            continue
    
    st.warning("⚠️ Gemini 로드 실패: AI 인사이트 없이 기본 분석 모드로 사용 가능합니다.")
    return None

llm = get_llm()

# --- 데이터 로드 ---
st.divider()
st.subheader(" 설비 데이터")

uploaded_file = st.file_uploader("CSV 파일 업로드", type=['csv'])

@st.cache_data
def load_data(file):
    """CSV 로드 및 전처리"""
    try:
        df = pd.read_csv(file, skipinitialspace=True, encoding='utf-8')
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        
        # 날짜 변환
        for col in df.columns:
            if 'date' in col or 'dt' in col or 'dtm' in col:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except Exception as e:
                    log_error("DateConversionError", f"{col} 날짜 변환 실패", str(e))
        
        # 숫자 변환
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    converted = pd.to_numeric(df[col], errors='coerce')
                    if not converted.isna().all():
                        df[col] = converted
                except Exception as e:
                    log_error("NumericConversionError", f"{col} 숫자 변환 실패", str(e))
        
        df.dropna(how='all', inplace=True)
        st.success(f" 로드 완료: {len(df):,}행 × {len(df.columns)}컬럼")
        return df
    except Exception as e:
        log_error("DataLoadError", "CSV 로딩 실패", str(e))
        st.error(f" 로딩 오류: {e}")
        return None

df_facility = None

if uploaded_file:
    df_facility = load_data(uploaded_file)
else:
    st.info(" CSV 파일을 업로드하세요")
    
    if st.button(" 샘플 데이터로 테스트"):
        import numpy as np
        np.random.seed(42)
        
        dates = pd.date_range('2024-01-01', periods=300, freq='D')  # 300일
        
        prod_wgt_values = []
        for date in dates:
            month = date.month
            base_value = 1500 + (month - 6) * 50
            if np.random.random() > 0.95:
                prod_wgt_values.append(base_value + np.random.uniform(1000, 2000))
            else:
                prod_wgt_values.append(np.random.normal(base_value, 200))
        
        df_facility = pd.DataFrame({
            'wrk_date': dates,
            'md_shft': np.random.choice(['A', 'B', 'C'], 300),
            'prod_wgt': prod_wgt_values,
            'eaf_wat_sum': np.random.normal(5000, 1000, 300),
            'wat_unit': np.random.normal(450, 50, 300)
        })
        st.success(" 샘플 데이터 생성 (300일, 이상치 포함)!")
        st.rerun()

if df_facility is not None:
    with st.expander(" 데이터 미리보기"):
        st.dataframe(df_facility.head(10))
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"** 기본 정보:**")
        st.write(f"- 행 수: {len(df_facility):,}")
        st.write(f"- 컬럼 수: {len(df_facility.columns)}")
        st.write(f"- 메모리: {df_facility.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    with col2:
        numeric_cols = df_facility.select_dtypes(include=['int64', 'float64']).columns.tolist()
        st.write(f"** 데이터 타입:**")
        st.write(f"- 수치형: {len(numeric_cols)}개")
        st.write(f"- 범주형: {len(df_facility.select_dtypes(include=['object']).columns)}개")
    
    with st.expander(" 컬럼 상세 정보"):
        col_info = pd.DataFrame({
            '컬럼명': df_facility.columns,
            '데이터타입': df_facility.dtypes.values,
            '결측치': df_facility.isnull().sum().values,
            '고유값수': [df_facility[col].nunique() for col in df_facility.columns]
        })
        st.dataframe(col_info)
    
    if numeric_cols:
        with st.expander(" 수치형 컬럼 통계"):
            st.dataframe(df_facility[numeric_cols].describe().T)
    
    # --- 데이터 전처리 옵션 ---
    st.divider()
    st.subheader(" 데이터 전처리 설정")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        use_outlier_removal = st.checkbox(" 이상치 제거 사용", value=False, 
                                          help="통계적 방법으로 이상치/노이즈를 제거합니다")
    
    with col2:
        if use_outlier_removal:
            outlier_method = st.selectbox(
                "제거 방법",
                ["iqr", "zscore", "percentile"],
                format_func=lambda x: {
                    "iqr": "IQR (사분위수) - 추천",
                    "zscore": "Z-Score (표준편차)",
                    "percentile": "Percentile (백분위수)"
                }[x],
                help="IQR: 가장 일반적, Z-Score: 정규분포 데이터, Percentile: 극단값 제거"
            )
        else:
            outlier_method = "iqr"
    
    with col3:
        if use_outlier_removal:
            if outlier_method == "iqr":
                outlier_threshold = st.slider("IQR 배수", 1.0, 3.0, 1.5, 0.1,
                                             help="1.5: 표준, 2.0: 관대, 1.0: 엄격")
            elif outlier_method == "zscore":
                outlier_threshold = st.slider("Z-Score 임계값", 2.0, 4.0, 3.0, 0.5,
                                             help="3.0: 표준 (99.7%), 2.0: 엄격")
            else:
                outlier_threshold = st.slider("제거 백분위수 (%)", 0.5, 5.0, 1.0, 0.5,
                                             help="상하위 몇 %를 제거할지")
        else:
            outlier_threshold = 1.5
    
    # --- AI 질의응답 ---
    st.divider()
    st.subheader(" AI에게 질문하기")
    
    # === 필터 섹션 추가 ===
    with st.expander(" 데이터 필터 (선택사항)", expanded=False):
        st.caption("⚠️ 필터를 적용하면 선택한 조건에 맞는 데이터만 분석됩니다")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 날짜 필터
            st.markdown("** 기간 필터**")
            date_cols = [col for col in df_facility.columns if 'date' in col.lower() or 'wrk_date' in col.lower()]
            
            if date_cols:
                date_col = date_cols[0]
                df_facility[date_col] = pd.to_datetime(df_facility[date_col], errors='coerce')
                
                min_date = df_facility[date_col].min()
                max_date = df_facility[date_col].max()
                
                filter_start_date = st.date_input(
                    "시작날짜",
                    value=min_date,
                    min_value=min_date,
                    max_value=max_date,
                    key="filter_start_date"
                )
                
                filter_end_date = st.date_input(
                    "종료날짜",
                    value=max_date,
                    min_value=min_date,
                    max_value=max_date,
                    key="filter_end_date"
                )
            else:
                st.info(" 날짜 컬럼이 없습니다")
                filter_start_date = None
                filter_end_date = None
        
        with col2:
            # 분류 필터 (대/중/소)
            st.markdown("**️ 분류 필터**")
            
            # 대분류
            if 'irn_larg_nm' in df_facility.columns:
                larg_options = ['전체'] + sorted(df_facility['irn_larg_nm'].dropna().unique().tolist())
                filter_larg = st.selectbox("대분류 (irn_larg_nm)", larg_options, key="filter_larg")
            else:
                filter_larg = '전체'
                st.caption(" irn_larg_nm 컬럼이 없습니다")
            
            # 중분류 (대분류에 종속)
            if 'irn_mid_nm' in df_facility.columns:
                if filter_larg != '전체':
                    mid_filtered = df_facility[df_facility['irn_larg_nm'] == filter_larg]['irn_mid_nm'].dropna().unique()
                    mid_options = ['전체'] + sorted(mid_filtered.tolist())
                else:
                    mid_options = ['전체'] + sorted(df_facility['irn_mid_nm'].dropna().unique().tolist())
                filter_mid = st.selectbox("중분류 (irn_mid_nm)", mid_options, key="filter_mid")
            else:
                filter_mid = '전체'
                st.caption(" irn_mid_nm 컬럼이 없습니다")
            
            # 소분류 (중분류에 종속)
            if 'irn_sml_nm' in df_facility.columns:
                if filter_mid != '전체':
                    if filter_larg != '전체':
                        sml_filtered = df_facility[
                            (df_facility['irn_larg_nm'] == filter_larg) &
                            (df_facility['irn_mid_nm'] == filter_mid)
                        ]['irn_sml_nm'].dropna().unique()
                    else:
                        sml_filtered = df_facility[df_facility['irn_mid_nm'] == filter_mid]['irn_sml_nm'].dropna().unique()
                    sml_options = ['전체'] + sorted(sml_filtered.tolist())
                else:
                    sml_options = ['전체'] + sorted(df_facility['irn_sml_nm'].dropna().unique().tolist())
                filter_sml = st.selectbox("소분류 (irn_sml_nm)", sml_options, key="filter_sml")
            else:
                filter_sml = '전체'
                st.caption(" irn_sml_nm 컬럼이 없습니다")
        
        # 필터 적용 버튼
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn1:
            if st.button(" 필터 적용", type="primary", use_container_width=True):
                df_filtered = df_facility.copy()
                
                # 날짜 필터
                if filter_start_date and filter_end_date and date_cols:
                    df_filtered = df_filtered[
                        (df_filtered[date_col] >= pd.Timestamp(filter_start_date)) &
                        (df_filtered[date_col] <= pd.Timestamp(filter_end_date))
                    ]
                
                # 분류 필터
                if filter_larg != '전체' and 'irn_larg_nm' in df_filtered.columns:
                    df_filtered = df_filtered[df_filtered['irn_larg_nm'] == filter_larg]
                if filter_mid != '전체' and 'irn_mid_nm' in df_filtered.columns:
                    df_filtered = df_filtered[df_filtered['irn_mid_nm'] == filter_mid]
                if filter_sml != '전체' and 'irn_sml_nm' in df_filtered.columns:
                    df_filtered = df_filtered[df_filtered['irn_sml_nm'] == filter_sml]
                
                st.session_state.df_filtered = df_filtered
                st.session_state.filter_applied = True
                st.success(f" 필터 적용 완료: {len(df_filtered):,}행 (원본: {len(df_facility):,}행)")
        
        with col_btn2:
            if st.button(" 필터 초기화", use_container_width=True):
                if 'df_filtered' in st.session_state:
                    del st.session_state.df_filtered
                if 'filter_applied' in st.session_state:
                    del st.session_state.filter_applied
                st.success(" 필터가 초기화되었습니다")
                st.rerun()
    
    # 필터된 데이터 사용
    if 'df_filtered' in st.session_state and st.session_state.get('filter_applied', False):
        df_work = st.session_state.df_filtered
        st.info(f" 필터링된 데이터 사용 중: {len(df_work):,}행")
    else:
        df_work = df_facility
    
    sample_qs = {
        "Q1": "prod_wgt 평균은?",
        "Q2": "prod_wgt 일별 추이",
        "Q3": "md_shft별 prod_wgt 일별 추이",
        "Q4": "prod_wgt 1월부터 6월까지 월별 추이",
        "Q5": "md_shft별 prod_wgt 월별 막대그래프"
    }
    
    st.write("** 샘플 질문:**")
    cols = st.columns(5)
    for idx, (key, q) in enumerate(sample_qs.items()):
        if cols[idx].button(key, help=q, key=f"sample_q_{idx}"):
            st.session_state.sample_question = q
    
    user_question = st.text_input(
        "질문:",
        value=st.session_state.get('sample_question', ''),
        placeholder="예: md_shft별로 prod_wgt 1월부터 8월까지 월별 막대그래프"
    )
    
    if st.button(" 분석", type="primary"):
        if user_question:
            try:
                # === 질문 분석 (개선된 로직) ===
                user_question_lower = user_question.lower()
                
                # 그래프 타입 키워드 (확장)
                line_keywords = ["선그래프", "라인", "line", "추이", "변화", "트렌드", "시계열"]
                bar_keywords = ["막대그래프", "막대", "bar", "바차트", "바그래프", "바"]
                pie_keywords = ["파이차트", "pie", "파이", "원그래프", "비율", "구성"]
                scatter_keywords = ["산점도", "scatter", "점그래프", "분산도"]
                area_keywords = ["영역차트", "area", "면적그래프", "영역"]
                box_keywords = ["박스플롯", "box", "상자그림", "boxplot"]
                histogram_keywords = ["히스토그램", "histogram", "분포도", "분포"]
                
                graph_keywords = (["그래프", "차트", "표현", "그려", "시각화"] + 
                                 line_keywords + bar_keywords + pie_keywords + 
                                 scatter_keywords + area_keywords + box_keywords + histogram_keywords)
                
                # 그래프 타입 감지
                has_line = any(kw in user_question_lower for kw in line_keywords)
                has_bar = any(kw in user_question_lower for kw in bar_keywords)
                has_pie = any(kw in user_question_lower for kw in pie_keywords)
                has_scatter = any(kw in user_question_lower for kw in scatter_keywords)
                has_area = any(kw in user_question_lower for kw in area_keywords)
                has_box = any(kw in user_question_lower for kw in box_keywords)
                has_histogram = any(kw in user_question_lower for kw in histogram_keywords)
                
                # 우선순위로 차트 타입 결정
                if has_bar:
                    chart_type = "bar"
                    chart_type_kr = "막대그래프"
                    detected_chart_reason = f"질문에 '{[kw for kw in bar_keywords if kw in user_question_lower][0]}' 키워드 발견"
                elif has_pie:
                    chart_type = "pie"
                    chart_type_kr = "파이차트"
                    detected_chart_reason = f"질문에 '{[kw for kw in pie_keywords if kw in user_question_lower][0]}' 키워드 발견"
                elif has_scatter:
                    chart_type = "scatter"
                    chart_type_kr = "산점도"
                    detected_chart_reason = f"질문에 '{[kw for kw in scatter_keywords if kw in user_question_lower][0]}' 키워드 발견"
                elif has_area:
                    chart_type = "area"
                    chart_type_kr = "영역차트"
                    detected_chart_reason = f"질문에 '{[kw for kw in area_keywords if kw in user_question_lower][0]}' 키워드 발견"
                elif has_box:
                    chart_type = "box"
                    chart_type_kr = "박스플롯"
                    detected_chart_reason = f"질문에 '{[kw for kw in box_keywords if kw in user_question_lower][0]}' 키워드 발견"
                elif has_histogram:
                    chart_type = "histogram"
                    chart_type_kr = "히스토그램"
                    detected_chart_reason = f"질문에 '{[kw for kw in histogram_keywords if kw in user_question_lower][0]}' 키워드 발견"
                elif has_line:
                    chart_type = "line"
                    chart_type_kr = "선그래프"
                    detected_chart_reason = f"질문에 '{[kw for kw in line_keywords if kw in user_question_lower][0]}' 키워드 발견"
                else:
                    # 기본값: 선그래프
                    chart_type = "line"
                    chart_type_kr = "선그래프"
                    detected_chart_reason = "키워드 없음 - 기본값 사용"
                
                # ⭐ 시간 단위 키워드 명확히 분리
                daily_keywords = ["일별", "날짜별", "daily", "day by day"]
                monthly_keywords = ["월별", "monthly", "month by month"]
                
                # 키워드 감지 (정확히)
                has_daily = any(kw in user_question_lower for kw in daily_keywords)
                has_monthly = any(kw in user_question_lower for kw in monthly_keywords)
                
                # 우선순위: 명시적 키워드 > 기본값(일별)
                if has_monthly and not has_daily:
                    # 월별만 있음
                    time_unit = "month"
                    time_unit_kr = "월별"
                    detected_reason = f"질문에 '{[kw for kw in monthly_keywords if kw in user_question_lower][0]}' 키워드 발견"
                elif has_daily:
                    # 일별 있음 (또는 둘 다 있으면 일별 우선)
                    time_unit = "day"
                    time_unit_kr = "일별"
                    detected_reason = f"질문에 '{[kw for kw in daily_keywords if kw in user_question_lower][0]}' 키워드 발견"
                else:
                    # 키워드 없음 - 기본값 일별
                    time_unit = "day"
                    time_unit_kr = "일별"
                    detected_reason = "키워드 없음 - 기본값 사용"
                
                time_keywords = daily_keywords + monthly_keywords + ["추이", "변화", "시계열"]
                is_time_series = any(kw in user_question_lower for kw in time_keywords)
                
                multi_keywords = ["계열", "조로", "구분", "별로", "분리", "그룹별", "나누어", "각각"]
                
                wants_graph = any(kw in user_question_lower for kw in graph_keywords)
                is_multi_series = any(kw in user_question_lower for kw in multi_keywords)
                
                # 감지 결과 표시
                if wants_graph or is_time_series:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f" 감지된 그래프 타입: **{chart_type_kr}** ({detected_chart_reason})")
                    with col2:
                        st.info(f" 감지된 시간 단위: **{time_unit_kr}** ({detected_reason})")
                st.info(f" 감지된 그래프 타입: **{chart_type_kr}**")
                
                # 날짜 컬럼 찾기
                date_col = None
                for col in df_work.columns:
                    if 'date' in col.lower() and pd.api.types.is_datetime64_any_dtype(df_work[col]):
                        date_col = col
                        break
                
                # 파이차트가 아니고 시계열 분석일 때만 날짜 컬럼 필수
                if not date_col and (wants_graph or is_time_series) and chart_type != "pie":
                    st.error(" 날짜 컬럼을 찾을 수 없습니다. 컬럼명에 'date'가 포함되어야 합니다.")
                    log_error("ColumnNotFound", "날짜 컬럼 없음", f"사용 가능 컬럼: {df_work.columns.tolist()}")
                
                # 분석 컬럼 찾기
                mentioned_col = None
                for col in numeric_cols:
                    if col in user_question_lower:
                        mentioned_col = col
                        break
                
                # 파이차트가 아닐 때만 수치 컬럼 필수
                if not mentioned_col and (wants_graph or is_time_series) and chart_type != "pie":
                    st.error(f" 분석할 수치 컬럼을 찾을 수 없습니다. 질문에 컬럼명을 포함해주세요: {', '.join(numeric_cols)}")
                    log_error("ColumnNotFound", "수치 컬럼 없음", f"질문: {user_question}")
                
                # 그룹 컬럼 찾기
                group_col = None
                if is_multi_series:
                    cat_cols = df_work.select_dtypes(include=['object']).columns
                    for col in cat_cols:
                        if col in user_question_lower:
                            group_col = col
                            break
                
                # === 우선순위 1: 그래프 요청 ===
                if (wants_graph or is_time_series) and date_col and mentioned_col:
                    
                    # 데이터 복사
                    temp_df = df_work.copy()
                    
                    # 이상치 제거
                    outlier_info = None
                    removed_count = 0
                    if use_outlier_removal:
                        st.info(f" 이상치 제거 중... (방법: {outlier_method}, 임계값: {outlier_threshold})")
                        temp_df, removed_count, outlier_info = remove_outliers(
                            temp_df, mentioned_col, outlier_method, outlier_threshold
                        )
                        
                        if removed_count > 0:
                            st.success(f" 이상치 제거 완료: {removed_count:,}개 행 제거 ({removed_count/len(df_work)*100:.1f}%)")
                        else:
                            st.info(" 제거된 이상치가 없습니다.")
                    
                    # ⭐ 시간 단위에 따라 그룹화
                    if time_unit == "day":
                        temp_df['time_group'] = temp_df[date_col].dt.date
                        x_label = "날짜"
                    else:  # month
                        temp_df['time_group'] = temp_df[date_col].dt.month
                        x_label = "월"
                    
                    # === 범위 필터링 (개선) ===
                    import re
                    
                    range_filtered = False
                    
                    if time_unit == "month":
                        # 월 범위 패턴 감지
                        # 패턴 1: "1월부터 8월까지", "1월에서 8월", "1월~8월"
                        month_range_patterns = [
                            r'(\d{1,2})월?\s*(?:부터|에서|~|-)\s*(\d{1,2})월?(?:까지)?',
                            r'(\d{1,2})\s*~\s*(\d{1,2})월',
                            r'(\d{1,2})-(\d{1,2})월'
                        ]
                        
                        for pattern in month_range_patterns:
                            match = re.search(pattern, user_question)
                            if match:
                                start_month = int(match.group(1))
                                end_month = int(match.group(2))
                                
                                if 1 <= start_month <= 12 and 1 <= end_month <= 12:
                                    temp_df = temp_df[temp_df['time_group'].between(start_month, end_month)]
                                    range_filtered = True
                                    st.success(f" 범위 필터링: {start_month}월 ~ {end_month}월")
                                    break
                    
                    elif time_unit == "day":
                        # 날짜 범위 패턴 감지
                        # 패턴 1: "2024-01-01부터 2024-08-31까지"
                        date_range_pattern = r'(\d{4}-\d{2}-\d{2})\s*(?:부터|에서|~|-)\s*(\d{4}-\d{2}-\d{2})(?:까지)?'
                        match = re.search(date_range_pattern, user_question)
                        
                        if match:
                            try:
                                start_date = pd.to_datetime(match.group(1)).date()
                                end_date = pd.to_datetime(match.group(2)).date()
                                
                                temp_df = temp_df[(temp_df['time_group'] >= start_date) & 
                                                 (temp_df['time_group'] <= end_date)]
                                range_filtered = True
                                st.success(f" 범위 필터링: {start_date} ~ {end_date}")
                            except:
                                pass
                        
                        # 패턴 2: "1월 1일부터 8월 31일까지" (간단한 버전)
                        if not range_filtered:
                            simple_date_pattern = r'(\d{1,2})월\s*(\d{1,2})일\s*(?:부터|에서)?\s*(?:~|-)?\s*(\d{1,2})월\s*(\d{1,2})일'
                            match = re.search(simple_date_pattern, user_question)
                            
                            if match:
                                try:
                                    start_month = int(match.group(1))
                                    start_day = int(match.group(2))
                                    end_month = int(match.group(3))
                                    end_day = int(match.group(4))
                                    
                                    # 현재 년도 사용
                                    current_year = temp_df[date_col].dt.year.iloc[0]
                                    start_date = pd.Timestamp(year=current_year, month=start_month, day=start_day).date()
                                    end_date = pd.Timestamp(year=current_year, month=end_month, day=end_day).date()
                                    
                                    temp_df = temp_df[(temp_df['time_group'] >= start_date) & 
                                                     (temp_df['time_group'] <= end_date)]
                                    range_filtered = True
                                    st.success(f" 범위 필터링: {start_date} ~ {end_date}")
                                except:
                                    pass
                    
                    if is_multi_series and group_col:
                        # === 다중 계열 분석 ===
                        st.markdown(f"###  계열별 {time_unit_kr} {chart_type_kr}")
                        
                        if use_outlier_removal and removed_count > 0:
                            st.caption(f" 이상치 제거 적용됨: {removed_count:,}개 데이터 포인트 제거")
                        
                        multi = temp_df.groupby(['time_group', group_col])[mentioned_col].mean().reset_index()
                        multi.columns = [x_label, group_col, mentioned_col]
                        
                        # 고유 key로 차트 생성
                        chart_key = f"chart_{uuid.uuid4().hex[:8]}"
                        chart_title = f'{mentioned_col}의 {group_col}별 {time_unit_kr} 평균{"(이상치 제거)" if use_outlier_removal else ""}'
                        
                        # 차트 타입에 따라 다른 그래프 생성
                        if chart_type == "bar":
                            fig = px.bar(multi, x=x_label, y=mentioned_col, color=group_col,
                                        title=chart_title, barmode='group')
                        elif chart_type == "area":
                            fig = px.area(multi, x=x_label, y=mentioned_col, color=group_col,
                                         title=chart_title)
                        elif chart_type == "scatter":
                            fig = px.scatter(multi, x=x_label, y=mentioned_col, color=group_col,
                                           title=chart_title, size_max=10)
                        elif chart_type == "box":
                            # 박스플롯은 원본 데이터 필요
                            fig = px.box(temp_df, x='time_group', y=mentioned_col, color=group_col,
                                        title=chart_title)
                        else:  # line (기본값)
                            fig = px.line(multi, x=x_label, y=mentioned_col, color=group_col,
                                        markers=True, title=chart_title)
                        
                        fig.update_xaxes(title=x_label)
                        fig.update_yaxes(title=f"{mentioned_col} 평균")
                        fig.update_layout(legend_title=group_col, height=500)
                        st.plotly_chart(fig, use_container_width=True, key=chart_key)
                        
                        with st.expander(" 계열별 데이터 테이블"):
                            pivot = multi.pivot(index=x_label, columns=group_col, values=mentioned_col)
                            st.dataframe(pivot)
                            
                            st.markdown("---")
                            st.markdown("####  데이터 처리 프로세스")
                            
                            process_steps = f"""
**1단계: 원본 데이터 로드**
- 파일: CSV 업로드 또는 샘플 데이터
- 행 수: {len(df_work):,}개
- 날짜 컬럼: `{date_col}`
- 분석 컬럼: `{mentioned_col}`
- 그룹 컬럼: `{group_col}`

**2단계: 이상치 제거** {' 적용됨' if use_outlier_removal else ' 적용 안 됨'}
{f"- 방법: {outlier_method}" if use_outlier_removal else ""}
{f"- 제거된 행: {removed_count:,}개 ({removed_count/len(df_work)*100:.1f}%)" if use_outlier_removal and removed_count > 0 else ""}
{f"- 남은 행: {len(temp_df):,}개" if use_outlier_removal else ""}

**3단계: 시간 단위 변환**
- 입력: 날짜 컬럼 (`{date_col}`)
- 변환: {time_unit_kr} 단위로 그룹화
- 결과: `time_group` 컬럼 생성

**4단계: 그룹별 집계**
- 그룹: `time_group` + `{group_col}`
- 집계 방법: 평균 (mean)
- 집계 컬럼: `{mentioned_col}`
- 결과 행 수: {len(multi):,}개

**5단계: 피벗 테이블 생성**
- 인덱스: {x_label}
- 컬럼: {group_col}
- 값: {mentioned_col} 평균
- 최종 크기: {len(pivot)} 행 × {len(pivot.columns)} 열
"""
                            st.markdown(process_steps)
                            
                            st.markdown("####  데이터 처리 코드")
                            
                            data_code = f"""import pandas as pd

# 1단계: 원본 데이터 로드
df = pd.read_csv('your_file.csv')
print(f"원본 데이터: {{len(df):,}}행")

# 날짜 컬럼 변환
df['{date_col}'] = pd.to_datetime(df['{date_col}'])
"""
                            
                            if use_outlier_removal:
                                if outlier_method == 'iqr':
                                    data_code += f"""
# 2단계: IQR 방법으로 이상치 제거
Q1 = df['{mentioned_col}'].quantile(0.25)
Q3 = df['{mentioned_col}'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - {outlier_threshold} * IQR
upper_bound = Q3 + {outlier_threshold} * IQR

df_clean = df[(df['{mentioned_col}'] >= lower_bound) & 
              (df['{mentioned_col}'] <= upper_bound)].copy()
print(f"이상치 제거 후: {{len(df_clean):,}}행 ({{len(df) - len(df_clean):,}}개 제거)")
"""
                                elif outlier_method == 'zscore':
                                    data_code += f"""
# 2단계: Z-Score 방법으로 이상치 제거
import numpy as np
mean = df['{mentioned_col}'].mean()
std = df['{mentioned_col}'].std()
z_scores = np.abs((df['{mentioned_col}'] - mean) / std)

df_clean = df[z_scores < {outlier_threshold}].copy()
print(f"이상치 제거 후: {{len(df_clean):,}}행 ({{len(df) - len(df_clean):,}}개 제거)")
"""
                                else:
                                    data_code += f"""
# 2단계: Percentile 방법으로 이상치 제거
lower_bound = df['{mentioned_col}'].quantile({outlier_threshold / 100})
upper_bound = df['{mentioned_col}'].quantile({(100 - outlier_threshold) / 100})

df_clean = df[(df['{mentioned_col}'] >= lower_bound) & 
              (df['{mentioned_col}'] <= upper_bound)].copy()
print(f"이상치 제거 후: {{len(df_clean):,}}행 ({{len(df) - len(df_clean):,}}개 제거)")
"""
                                data_code += f"""
df = df_clean  # 정제된 데이터 사용
"""
                            else:
                                data_code += f"""
# 2단계: 이상치 제거 안 함
df_clean = df.copy()
"""
                            
                            if time_unit == 'day':
                                data_code += f"""
# 3단계: 일별 단위로 변환
df['time_group'] = df['{date_col}'].dt.date
"""
                            else:
                                data_code += f"""
# 3단계: 월별 단위로 변환
df['time_group'] = df['{date_col}'].dt.month
"""
                            
                            data_code += f"""
# 4단계: 그룹별 평균 계산
grouped_data = df.groupby(['time_group', '{group_col}'])['{mentioned_col}'].mean().reset_index()
grouped_data.columns = ['{x_label}', '{group_col}', '{mentioned_col}']
print(f"그룹별 집계: {{len(grouped_data):,}}행")

# 5단계: 피벗 테이블 생성
pivot_table = grouped_data.pivot(index='{x_label}', 
                                  columns='{group_col}', 
                                  values='{mentioned_col}')
print(f"피벗 테이블: {{len(pivot_table)}}행 × {{len(pivot_table.columns)}}열")
print(pivot_table)
"""
                            
                            st.code(data_code, language="python")
                        
                        with st.expander(" 그래프 생성 코드"):
                            code = f"""import plotly.express as px
import pandas as pd

# 데이터 준비
data = {multi.to_dict('list')}
df = pd.DataFrame(data)

# 그래프 생성
fig = px.line(df, 
              x='{x_label}', 
              y='{mentioned_col}', 
              color='{group_col}',
              markers=True, 
              title='{mentioned_col}의 {group_col}별 {time_unit_kr} 평균 추이{"(이상치 제거)" if use_outlier_removal else ""}')

fig.update_xaxes(title='{x_label}')
fig.update_yaxes(title='{mentioned_col} 평균')
fig.update_layout(legend_title='{group_col}', height=500)

fig.show()"""
                            st.code(code, language="python")
                        
                        # 계열별 인사이트
                        st.markdown("###  계열별 핵심 인사이트")
                        
                        insights_text = ""
                        for group in sorted(temp_df[group_col].unique()):
                            group_data = multi[multi[group_col] == group]
                            if len(group_data) > 0:
                                max_time = group_data.loc[group_data[mentioned_col].idxmax(), x_label]
                                max_value = group_data[mentioned_col].max()
                                min_time = group_data.loc[group_data[mentioned_col].idxmin(), x_label]
                                min_value = group_data[mentioned_col].min()
                                avg_value = group_data[mentioned_col].mean()
                                
                                insight = f"""
**{group_col} = {group}**
- 최고점: {max_time} ({max_value:,.2f})
- 최저점: {min_time} ({min_value:,.2f})
- 평균: {avg_value:,.2f}
- 변동폭: {max_value - min_value:,.2f} ({((max_value/min_value - 1) * 100):.1f}% 증가)
                                """
                                st.info(insight)
                                insights_text += insight + "\n"
                        
                        # AI 인사이트
                        if llm:
                            with st.spinner("AI 인사이트 생성 중..."):
                                try:
                                    prompt = f"""
다음은 {group_col}별 {mentioned_col}의 {time_unit_kr} 평균 데이터입니다{"(이상치 제거 후)" if use_outlier_removal else ""}:
{multi.to_string()}

철강 설비 데이터 전문가로서, 각 그룹별로 주목할 만한 특징과 차이점을 한국어로 3-4가지 설명하세요.
"""
                                    insight = llm.invoke(prompt)
                                    ai_insight = insight.content
                                    st.success(f"** AI 인사이트:**\n\n{ai_insight}")
                                    insights_text += f"\n AI 분석:\n{ai_insight}"
                                except Exception as e:
                                    log_error("AIInsightError", "AI 인사이트 생성 실패", str(e))
                                    st.warning(f"⚠️ AI 인사이트 생성 실패: {e}")
                        
                        # 히스토리에 추가
                        if chart_type == "bar":
                            multi_code = f"""import plotly.express as px
import pandas as pd

# 데이터 준비
data = {multi.to_dict('list')}
df = pd.DataFrame(data)

# 막대그래프 생성
fig = px.bar(df, 
             x='{x_label}', 
             y='{mentioned_col}', 
             color='{group_col}',
             title='{mentioned_col}의 {group_col}별 {time_unit_kr} 평균{"(이상치 제거)" if use_outlier_removal else ""}',
             barmode='group')

fig.update_xaxes(title='{x_label}')
fig.update_yaxes(title='{mentioned_col} 평균')
fig.update_layout(legend_title='{group_col}', height=500)

fig.show()"""
                        elif chart_type == "area":
                            multi_code = f"""import plotly.express as px
import pandas as pd

# 데이터 준비
data = {multi.to_dict('list')}
df = pd.DataFrame(data)

# 영역차트 생성
fig = px.area(df, 
              x='{x_label}', 
              y='{mentioned_col}', 
              color='{group_col}',
              title='{mentioned_col}의 {group_col}별 {time_unit_kr} 평균{"(이상치 제거)" if use_outlier_removal else ""}')

fig.update_xaxes(title='{x_label}')
fig.update_yaxes(title='{mentioned_col} 평균')
fig.update_layout(legend_title='{group_col}', height=500)

fig.show()"""
                        else:  # line (기본값)
                            multi_code = f"""import plotly.express as px
import pandas as pd

# 데이터 준비
data = {multi.to_dict('list')}
df = pd.DataFrame(data)

# 선그래프 생성
fig = px.line(df, 
              x='{x_label}', 
              y='{mentioned_col}', 
              color='{group_col}',
              markers=True, 
              title='{mentioned_col}의 {group_col}별 {time_unit_kr} 평균{"(이상치 제거)" if use_outlier_removal else ""}')

fig.update_xaxes(title='{x_label}')
fig.update_yaxes(title='{mentioned_col} 평균')
fig.update_layout(legend_title='{group_col}', height=500)

fig.show()"""
                        
                        # 데이터 처리 코드 생성
                        multi_data_code = f"""import pandas as pd

# 1단계: 원본 데이터 로드
df = pd.read_csv('your_file.csv')
print(f"원본 데이터: {{len(df):,}}행")

# 날짜 컬럼 변환
df['{date_col}'] = pd.to_datetime(df['{date_col}'])
"""
                        
                        if use_outlier_removal:
                            if outlier_method == 'iqr':
                                multi_data_code += f"""
# 2단계: IQR 방법으로 이상치 제거
Q1 = df['{mentioned_col}'].quantile(0.25)
Q3 = df['{mentioned_col}'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - {outlier_threshold} * IQR
upper_bound = Q3 + {outlier_threshold} * IQR

df_clean = df[(df['{mentioned_col}'] >= lower_bound) & 
              (df['{mentioned_col}'] <= upper_bound)].copy()
print(f"이상치 제거 후: {{len(df_clean):,}}행 ({{len(df) - len(df_clean):,}}개 제거)")

df = df_clean
"""
                            elif outlier_method == 'zscore':
                                multi_data_code += f"""
# 2단계: Z-Score 방법으로 이상치 제거
import numpy as np
mean = df['{mentioned_col}'].mean()
std = df['{mentioned_col}'].std()
z_scores = np.abs((df['{mentioned_col}'] - mean) / std)

df_clean = df[z_scores < {outlier_threshold}].copy()
print(f"이상치 제거 후: {{len(df_clean):,}}행 ({{len(df) - len(df_clean):,}}개 제거)")

df = df_clean
"""
                            else:
                                multi_data_code += f"""
# 2단계: Percentile 방법으로 이상치 제거
lower_bound = df['{mentioned_col}'].quantile({outlier_threshold / 100})
upper_bound = df['{mentioned_col}'].quantile({(100 - outlier_threshold) / 100})

df_clean = df[(df['{mentioned_col}'] >= lower_bound) & 
              (df['{mentioned_col}'] <= upper_bound)].copy()
print(f"이상치 제거 후: {{len(df_clean):,}}행 ({{len(df) - len(df_clean):,}}개 제거)")

df = df_clean
"""
                        
                        if time_unit == 'day':
                            multi_data_code += f"""
# 3단계: 일별 단위로 변환
df['time_group'] = df['{date_col}'].dt.date
"""
                        else:
                            multi_data_code += f"""
# 3단계: 월별 단위로 변환
df['time_group'] = df['{date_col}'].dt.month
"""
                        
                        multi_data_code += f"""
# 4단계: 그룹별 평균 계산
grouped_data = df.groupby(['time_group', '{group_col}'])['{mentioned_col}'].mean().reset_index()
grouped_data.columns = ['{x_label}', '{group_col}', '{mentioned_col}']
print(f"그룹별 집계: {{len(grouped_data):,}}행")

# 5단계: 피벗 테이블 생성
pivot_table = grouped_data.pivot(index='{x_label}', 
                                  columns='{group_col}', 
                                  values='{mentioned_col}')
print(f"피벗 테이블: {{len(pivot_table)}}행 × {{len(pivot_table.columns)}}열")
print(pivot_table)
"""
                        
                        # 결과를 세션에 임시 저장
                        st.session_state.last_analysis = {
                            'question': user_question,
                            'result_type': f"계열별_{time_unit_kr}_추이",
                            'figure': fig,
                            'data': multi,
                            'insights': insights_text,
                            'code': multi_code,
                            'data_code': multi_data_code,
                            'chart_type': chart_type_kr,
                            'time_unit': time_unit_kr
                        }
                        
                        # 저장 버튼
                        st.divider()
                        col_save1, col_save2, col_save3 = st.columns([2, 1, 2])
                        with col_save2:
                            if st.button(" 히스토리에 저장", type="primary", use_container_width=True, key="save_multi"):
                                add_to_full_history(
                                    question=st.session_state.last_analysis['question'],
                                    result_type=st.session_state.last_analysis['result_type'],
                                    figure=st.session_state.last_analysis['figure'],
                                    data=st.session_state.last_analysis['data'],
                                    insights=st.session_state.last_analysis['insights'],
                                    code=st.session_state.last_analysis['code'],
                                    data_code=st.session_state.last_analysis['data_code'],
                                    chart_type=st.session_state.last_analysis['chart_type'],
                                    time_unit=st.session_state.last_analysis['time_unit']
                                )
                                st.success(" 히스토리에 저장되었습니다!")
                                st.balloons()

                    
                    else:
                        # === 단일 계열 분석 ===
                        st.markdown(f"###  {time_unit_kr} {chart_type_kr}")
                        
                        if use_outlier_removal and removed_count > 0:
                            st.caption(f" 이상치 제거 적용됨: {removed_count:,}개 데이터 포인트 제거")
                        
                        time_data = temp_df.groupby('time_group')[mentioned_col].mean().reset_index()
                        time_data.columns = [x_label, mentioned_col]
                        
                        # 고유 key로 차트 생성
                        chart_key = f"chart_{uuid.uuid4().hex[:8]}"
                        chart_title = f'{mentioned_col}의 {time_unit_kr} 평균{"(이상치 제거)" if use_outlier_removal else ""}'
                        
                        # 차트 타입에 따라 다른 그래프 생성
                        if chart_type == "bar":
                            fig = px.bar(time_data, x=x_label, y=mentioned_col,
                                        title=chart_title)
                        elif chart_type == "area":
                            fig = px.area(time_data, x=x_label, y=mentioned_col,
                                         title=chart_title)
                        elif chart_type == "scatter":
                            fig = px.scatter(time_data, x=x_label, y=mentioned_col,
                                           title=chart_title, size_max=10)
                        elif chart_type == "pie":
                            # 파이차트는 시계열에 부적합하지만 요청 시 생성
                            fig = px.pie(time_data, names=x_label, values=mentioned_col,
                                        title=chart_title)
                        elif chart_type == "box":
                            # 박스플롯은 원본 데이터 필요
                            fig = px.box(temp_df, y=mentioned_col,
                                        title=chart_title)
                        elif chart_type == "histogram":
                            # 히스토그램은 분포 확인용
                            fig = px.histogram(temp_df, x=mentioned_col,
                                             title=f'{mentioned_col} 분포{"(이상치 제거)" if use_outlier_removal else ""}')
                        else:  # line (기본값)
                            fig = px.line(time_data, x=x_label, y=mentioned_col,
                                        markers=True, title=chart_title)
                        
                        fig.update_xaxes(title=x_label)
                        fig.update_yaxes(title=f"{mentioned_col} 평균")
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True, key=chart_key)
                        
                        with st.expander(" 데이터 테이블"):
                            st.dataframe(time_data)
                            
                            st.markdown("---")
                            st.markdown("####  데이터 처리 프로세스")
                            
                            process_steps = f"""
**1단계: 원본 데이터 로드**
- 파일: CSV 업로드 또는 샘플 데이터
- 행 수: {len(df_work):,}개
- 날짜 컬럼: `{date_col}`
- 분석 컬럼: `{mentioned_col}`

**2단계: 이상치 제거** {' 적용됨' if use_outlier_removal else ' 적용 안 됨'}
{f"- 방법: {outlier_method}" if use_outlier_removal else ""}
{f"- 제거된 행: {removed_count:,}개 ({removed_count/len(df_work)*100:.1f}%)" if use_outlier_removal and removed_count > 0 else ""}
{f"- 남은 행: {len(temp_df):,}개" if use_outlier_removal else ""}

**3단계: 시간 단위 변환**
- 입력: 날짜 컬럼 (`{date_col}`)
- 변환: {time_unit_kr} 단위로 그룹화
- 결과: `time_group` 컬럼 생성

**4단계: 시간별 집계**
- 그룹: `time_group`
- 집계 방법: 평균 (mean)
- 집계 컬럼: `{mentioned_col}`
- 결과 행 수: {len(time_data):,}개

**5단계: 최종 데이터**
- 컬럼: [{x_label}, {mentioned_col}]
- 크기: {len(time_data)} 행 × 2 열
"""
                            st.markdown(process_steps)
                            
                            st.markdown("####  데이터 처리 코드")
                            
                            data_code = f"""import pandas as pd

# 1단계: 원본 데이터 로드
df = pd.read_csv('your_file.csv')
print(f"원본 데이터: {{len(df):,}}행")

# 날짜 컬럼 변환
df['{date_col}'] = pd.to_datetime(df['{date_col}'])
"""
                            
                            if use_outlier_removal:
                                if outlier_method == 'iqr':
                                    data_code += f"""
# 2단계: IQR 방법으로 이상치 제거
Q1 = df['{mentioned_col}'].quantile(0.25)
Q3 = df['{mentioned_col}'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - {outlier_threshold} * IQR
upper_bound = Q3 + {outlier_threshold} * IQR

df_clean = df[(df['{mentioned_col}'] >= lower_bound) & 
              (df['{mentioned_col}'] <= upper_bound)].copy()
print(f"이상치 제거 후: {{len(df_clean):,}}행 ({{len(df) - len(df_clean):,}}개 제거)")
"""
                                elif outlier_method == 'zscore':
                                    data_code += f"""
# 2단계: Z-Score 방법으로 이상치 제거
import numpy as np
mean = df['{mentioned_col}'].mean()
std = df['{mentioned_col}'].std()
z_scores = np.abs((df['{mentioned_col}'] - mean) / std)

df_clean = df[z_scores < {outlier_threshold}].copy()
print(f"이상치 제거 후: {{len(df_clean):,}}행 ({{len(df) - len(df_clean):,}}개 제거)")
"""
                                else:
                                    data_code += f"""
# 2단계: Percentile 방법으로 이상치 제거
lower_bound = df['{mentioned_col}'].quantile({outlier_threshold / 100})
upper_bound = df['{mentioned_col}'].quantile({(100 - outlier_threshold) / 100})

df_clean = df[(df['{mentioned_col}'] >= lower_bound) & 
              (df['{mentioned_col}'] <= upper_bound)].copy()
print(f"이상치 제거 후: {{len(df_clean):,}}행 ({{len(df) - len(df_clean):,}}개 제거)")
"""
                                data_code += f"""
df = df_clean  # 정제된 데이터 사용
"""
                            else:
                                data_code += f"""
# 2단계: 이상치 제거 안 함
df_clean = df.copy()
"""
                            
                            if time_unit == 'day':
                                data_code += f"""
# 3단계: 일별 단위로 변환
df['time_group'] = df['{date_col}'].dt.date
"""
                            else:
                                data_code += f"""
# 3단계: 월별 단위로 변환
df['time_group'] = df['{date_col}'].dt.month
"""
                            
                            data_code += f"""
# 4단계: 시간별 평균 계산
time_data = df.groupby('time_group')['{mentioned_col}'].mean().reset_index()
time_data.columns = ['{x_label}', '{mentioned_col}']
print(f"시간별 집계: {{len(time_data):,}}행")
print(time_data)
"""
                            
                            st.code(data_code, language="python")
                        
                        with st.expander(" 그래프 생성 코드"):
                            if chart_type == "bar":
                                code = f"""import plotly.express as px
import pandas as pd

# 데이터 준비
data = {time_data.to_dict('list')}
df = pd.DataFrame(data)

# 막대그래프 생성
fig = px.bar(df, 
             x='{x_label}', 
             y='{mentioned_col}',
             title='{mentioned_col}의 {time_unit_kr} 평균{"(이상치 제거)" if use_outlier_removal else ""}')

fig.update_xaxes(title='{x_label}')
fig.update_yaxes(title='{mentioned_col} 평균')
fig.update_layout(height=500)

fig.show()"""
                            elif chart_type == "area":
                                code = f"""import plotly.express as px
import pandas as pd

# 데이터 준비
data = {time_data.to_dict('list')}
df = pd.DataFrame(data)

# 영역차트 생성
fig = px.area(df, 
              x='{x_label}', 
              y='{mentioned_col}',
              title='{mentioned_col}의 {time_unit_kr} 평균{"(이상치 제거)" if use_outlier_removal else ""}')

fig.update_xaxes(title='{x_label}')
fig.update_yaxes(title='{mentioned_col} 평균')
fig.update_layout(height=500)

fig.show()"""
                            elif chart_type == "pie":
                                code = f"""import plotly.express as px
import pandas as pd

# 데이터 준비
data = {time_data.to_dict('list')}
df = pd.DataFrame(data)

# 파이차트 생성
fig = px.pie(df, 
             names='{x_label}', 
             values='{mentioned_col}',
             title='{mentioned_col}의 {time_unit_kr} 구성{"(이상치 제거)" if use_outlier_removal else ""}')

fig.show()"""
                            else:  # line (기본값)
                                code = f"""import plotly.express as px
import pandas as pd

# 데이터 준비
data = {time_data.to_dict('list')}
df = pd.DataFrame(data)

# 선그래프 생성
fig = px.line(df, 
              x='{x_label}', 
              y='{mentioned_col}',
              markers=True, 
              title='{mentioned_col}의 {time_unit_kr} 평균{"(이상치 제거)" if use_outlier_removal else ""}')

fig.update_xaxes(title='{x_label}')
fig.update_yaxes(title='{mentioned_col} 평균')
fig.update_layout(height=500)

fig.show()"""
                            st.code(code, language="python")
                        
                        # 인사이트
                        max_time = time_data.loc[time_data[mentioned_col].idxmax(), x_label]
                        max_val = time_data[mentioned_col].max()
                        min_time = time_data.loc[time_data[mentioned_col].idxmin(), x_label]
                        min_val = time_data[mentioned_col].min()
                        avg_val = time_data[mentioned_col].mean()
                        
                        insights_text = f"""
** 핵심 인사이트:**
- 최고점: {max_time} ({max_val:,.2f})
- 최저점: {min_time} ({min_val:,.2f})
- 평균: {avg_val:,.2f}
- 변동폭: {max_val - min_val:,.2f} ({((max_val/min_val - 1) * 100):.1f}% 증가)
                        """
                        st.info(insights_text)
                        
                        # 히스토리에 추가 - chart_type 반영
                        if chart_type == "bar":
                            single_code = f"""import plotly.express as px
import pandas as pd

# 데이터 준비
data = {time_data.to_dict('list')}
df = pd.DataFrame(data)

# 막대그래프 생성
fig = px.bar(df, 
             x='{x_label}', 
             y='{mentioned_col}',
             title='{mentioned_col}의 {time_unit_kr} 평균{"(이상치 제거)" if use_outlier_removal else ""}')

fig.update_xaxes(title='{x_label}')
fig.update_yaxes(title='{mentioned_col} 평균')
fig.update_layout(height=500)

fig.show()"""
                        elif chart_type == "area":
                            single_code = f"""import plotly.express as px
import pandas as pd

# 데이터 준비
data = {time_data.to_dict('list')}
df = pd.DataFrame(data)

# 영역차트 생성
fig = px.area(df, 
              x='{x_label}', 
              y='{mentioned_col}',
              title='{mentioned_col}의 {time_unit_kr} 평균{"(이상치 제거)" if use_outlier_removal else ""}')

fig.update_xaxes(title='{x_label}')
fig.update_yaxes(title='{mentioned_col} 평균')
fig.update_layout(height=500)

fig.show()"""
                        else:  # line
                            single_code = f"""import plotly.express as px
import pandas as pd

# 데이터 준비
data = {time_data.to_dict('list')}
df = pd.DataFrame(data)

# 선그래프 생성
fig = px.line(df, 
              x='{x_label}', 
              y='{mentioned_col}',
              markers=True, 
              title='{mentioned_col}의 {time_unit_kr} 평균{"(이상치 제거)" if use_outlier_removal else ""}')

fig.update_xaxes(title='{x_label}')
fig.update_yaxes(title='{mentioned_col} 평균')
fig.update_layout(height=500)

fig.show()"""
                        
                        # 데이터 처리 코드 생성
                        single_data_code = f"""import pandas as pd

# 1단계: 원본 데이터 로드
df = pd.read_csv('your_file.csv')
print(f"원본 데이터: {{len(df):,}}행")

# 날짜 컬럼 변환
df['{date_col}'] = pd.to_datetime(df['{date_col}'])
"""
                        
                        if use_outlier_removal:
                            if outlier_method == 'iqr':
                                single_data_code += f"""
# 2단계: IQR 방법으로 이상치 제거
Q1 = df['{mentioned_col}'].quantile(0.25)
Q3 = df['{mentioned_col}'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - {outlier_threshold} * IQR
upper_bound = Q3 + {outlier_threshold} * IQR

df_clean = df[(df['{mentioned_col}'] >= lower_bound) & 
              (df['{mentioned_col}'] <= upper_bound)].copy()
print(f"이상치 제거 후: {{len(df_clean):,}}행 ({{len(df) - len(df_clean):,}}개 제거)")

df = df_clean
"""
                            elif outlier_method == 'zscore':
                                single_data_code += f"""
# 2단계: Z-Score 방법으로 이상치 제거
import numpy as np
mean = df['{mentioned_col}'].mean()
std = df['{mentioned_col}'].std()
z_scores = np.abs((df['{mentioned_col}'] - mean) / std)

df_clean = df[z_scores < {outlier_threshold}].copy()
print(f"이상치 제거 후: {{len(df_clean):,}}행 ({{len(df) - len(df_clean):,}}개 제거)")

df = df_clean
"""
                            else:
                                single_data_code += f"""
# 2단계: Percentile 방법으로 이상치 제거
lower_bound = df['{mentioned_col}'].quantile({outlier_threshold / 100})
upper_bound = df['{mentioned_col}'].quantile({(100 - outlier_threshold) / 100})

df_clean = df[(df['{mentioned_col}'] >= lower_bound) & 
              (df['{mentioned_col}'] <= upper_bound)].copy()
print(f"이상치 제거 후: {{len(df_clean):,}}행 ({{len(df) - len(df_clean):,}}개 제거)")

df = df_clean
"""
                        
                        if time_unit == 'day':
                            single_data_code += f"""
# 3단계: 일별 단위로 변환
df['time_group'] = df['{date_col}'].dt.date
"""
                        else:
                            single_data_code += f"""
# 3단계: 월별 단위로 변환
df['time_group'] = df['{date_col}'].dt.month
"""
                        
                        single_data_code += f"""
# 4단계: 시간별 평균 계산
time_data = df.groupby('time_group')['{mentioned_col}'].mean().reset_index()
time_data.columns = ['{x_label}', '{mentioned_col}']
print(f"시간별 집계: {{len(time_data):,}}행")
print(time_data)
"""
                        
                        # 결과를 세션에 임시 저장
                        st.session_state.last_analysis = {
                            'question': user_question,
                            'result_type': f"{time_unit_kr}_추이",
                            'figure': fig,
                            'data': time_data,
                            'insights': insights_text,
                            'code': single_code,
                            'data_code': single_data_code,
                            'chart_type': chart_type_kr,
                            'time_unit': time_unit_kr
                        }
                        
                        # 저장 버튼
                        st.divider()
                        col_save1, col_save2, col_save3 = st.columns([2, 1, 2])
                        with col_save2:
                            if st.button(" 히스토리에 저장", type="primary", use_container_width=True, key="save_single"):
                                add_to_full_history(
                                    question=st.session_state.last_analysis['question'],
                                    result_type=st.session_state.last_analysis['result_type'],
                                    figure=st.session_state.last_analysis['figure'],
                                    data=st.session_state.last_analysis['data'],
                                    insights=st.session_state.last_analysis['insights'],
                                    code=st.session_state.last_analysis['code'],
                                    data_code=st.session_state.last_analysis['data_code'],
                                    chart_type=st.session_state.last_analysis['chart_type'],
                                    time_unit=st.session_state.last_analysis['time_unit']
                                )
                                st.success(" 히스토리에 저장되었습니다!")
                                st.balloons()

                
                # === 우선순위 1.5: 파이차트 (시계열 아님) ===
                elif chart_type == "pie" and wants_graph:
                    st.markdown("###  파이차트 분석")
                    
                    # === 범위 기반 파이차트 감지 (대폭 개선) ===
                    import re
                    
                    # 범위 키워드 감지 (확장)
                    range_keywords = ['이하', '초과', '이상', '미만', '그룹', '나눠', '분류', '구분']
                    has_range_keyword = any(kw in user_question for kw in range_keywords)
                    
                    range_based = False
                    multi_range = False
                    threshold = None
                    all_numbers = []
                    
                    if has_range_keyword:
                        st.info(" 범위 키워드 감지!")
                        
                        # 숫자 추출 (연도 제외)
                        all_numbers = re.findall(r'\b(\d{1,4})\b', user_question)
                        all_numbers = [int(n) for n in all_numbers if 0 < int(n) < 10000 and int(n) != 2025 and int(n) != 2024]
                        st.info(f" 추출된 숫자: {all_numbers}")
                        
                        # 범위 감지 개선
                        if len(all_numbers) >= 1:
                            range_based = True
                            
                            # 다중 범위 패턴 감지
                            # 예: "400미만과 400이상" -> 2개 그룹
                            range_indicators = ['미만', '이하', '이상', '초과']
                            range_count = sum(1 for kw in range_indicators if kw in user_question)
                            
                            if range_count >= 2 or ('과' in user_question and any(kw in user_question for kw in range_indicators)):
                                multi_range = True
                                st.info(f" 다중 범위 감지: {range_count}개 조건, 경계값: {all_numbers}")
                            else:
                                threshold = all_numbers[0]
                                st.info(f" 단일 범위 감지: 기준값 {threshold}")
                        else:
                            st.warning("⚠️ 범위 키워드는 있지만 기준값을 찾을 수 없습니다.")
                    
                    # === 다중 범위 기반 파이차트 ===
                    if range_based and multi_range:
                        st.markdown("####  다중 범위 그룹 파이차트")
                        
                        # 수치형 컬럼 찾기
                        value_col = None
                        if mentioned_col:
                            value_col = mentioned_col
                        elif numeric_cols:
                            for col in numeric_cols:
                                if any(kw in col.lower() for kw in ['wgt', 'unit', 'cnt', 'count', 'sum', 'total']):
                                    value_col = col
                                    break
                            if not value_col:
                                value_col = numeric_cols[0]
                            st.info(f" 분석 컬럼: **{value_col}**")
                        
                        if value_col:
                            try:
                                # 경계값 설정
                                boundaries = sorted(set(all_numbers))
                                st.info(f" 경계값: {boundaries}")
                                
                                # 범위 기반 그룹 생성 함수 (개선)
                                def assign_group(value):
                                    if len(boundaries) == 1:
                                        # 단일 경계: 400 -> "400 미만", "400 이상"
                                        if '미만' in user_question:
                                            return f'{boundaries[0]} 미만' if value < boundaries[0] else f'{boundaries[0]} 이상'
                                        else:
                                            return f'{boundaries[0]} 이하' if value <= boundaries[0] else f'{boundaries[0]} 초과'
                                    
                                    elif len(boundaries) == 2:
                                        # 2개 경계: 400, 500 -> "400 미만", "400-500", "500 초과"
                                        if value < boundaries[0]:
                                            return f'{boundaries[0]} 미만'
                                        elif value < boundaries[1]:
                                            return f'{boundaries[0]}-{boundaries[1]}'
                                        else:
                                            return f'{boundaries[1]} 이상'
                                    
                                    else:
                                        # 3개 이상 경계
                                        for i in range(len(boundaries) - 1):
                                            if value < boundaries[i+1]:
                                                if i == 0:
                                                    return f'{boundaries[i+1]} 미만'
                                                else:
                                                    return f'{boundaries[i]}-{boundaries[i+1]}'
                                        return f'{boundaries[-1]} 이상'
                                
                                # 범위 그룹 생성
                                df_copy = df_work.copy()
                                df_copy['range_group'] = df_copy[value_col].apply(assign_group)
                                
                                # 그룹별 개수 계산
                                range_counts = df_copy['range_group'].value_counts().reset_index()
                                range_counts.columns = ['범위', '개수']
                                
                                # 비율 계산
                                range_counts['비율(%)'] = (range_counts['개수'] / range_counts['개수'].sum() * 100).round(2)
                                
                                # 파이차트 생성
                                fig = px.pie(
                                    range_counts,
                                    names='범위',
                                    values='개수',
                                    title=f'{value_col} 범위별 데이터 분포 ({", ".join(map(str, boundaries))} 기준)'
                                )
                                
                                fig.update_traces(
                                    textposition='inside',
                                    textinfo='percent+label+value',
                                    textfont_size=14
                                )
                                fig.update_layout(height=500)
                                
                                st.plotly_chart(fig, use_container_width=True, key=f"pie_multi_{uuid.uuid4().hex[:8]}")
                                
                                # 상세 통계
                                with st.expander(" 상세 통계"):
                                    st.dataframe(range_counts, use_container_width=True)
                                    
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("총 데이터 수", f"{range_counts['개수'].sum():,}")
                                    with col2:
                                        st.metric("그룹 수", len(range_counts))
                                
                                # 인사이트
                                insights_text = f"""**{value_col} 범위별 분포 분석**

** 경계값:** {', '.join(map(str, boundaries))}
** 그룹별 데이터:**

"""
                                for _, row in range_counts.iterrows():
                                    cnt = int(row['개수'])
                                    pct = round(row['비율(%)'], 1)
                                    insights_text += f"- {row['범위']}: {cnt}개 ({pct}%)\n"
                                
                                max_group = range_counts.iloc[0]
                                max_pct = round(max_group['비율(%)'], 1)
                                insights_text += f"\n가장 많은 그룹: {max_group['범위']} ({max_pct}%)"

                                
                                st.info(insights_text)
                                
                                # 코드 생성
                                range_data_code = f"""# 다중 범위 파이차트 - 데이터 처리
import pandas as pd

# 데이터 로드
df = pd.read_csv('your_file.csv')

# 범위 그룹 생성
boundaries = {boundaries}

def assign_group(value):
    if value < boundaries[0]:
        return f'{{boundaries[0]}} 미만'
    elif len(boundaries) == 1:
        return f'{{boundaries[0]}} 이상'
    # ... (추가 로직)

df['range_group'] = df['{value_col}'].apply(assign_group)

# 개수 집계
range_counts = df['range_group'].value_counts().reset_index()
range_counts.columns = ['범위', '개수']
print(range_counts)
"""
                                
                                range_code = f"""# 파이차트 생성
import plotly.express as px

fig = px.pie(
    range_counts,
    names='범위',
    values='개수',
    title='{value_col} 범위별 분포'
)
fig.update_traces(textposition='inside', textinfo='percent+label+value')
fig.show()
"""
                                
                                with st.expander(" 생성 코드"):
                                    st.code(range_data_code, language="python")
                                    st.code(range_code, language="python")
                                
                                # 세션에 저장
                                st.session_state.last_analysis = {
                                    'question': user_question,
                                    'result_type': '다중범위_파이차트',
                                    'figure': fig,
                                    'data': range_counts,
                                    'insights': insights_text,
                                    'code': range_code,
                                    'data_code': range_data_code,
                                    'chart_type': '파이차트',
                                    'time_unit': 'N/A'
                                }
                                
                                # 저장 버튼
                                st.divider()
                                col1, col2, col3 = st.columns([2, 1, 2])
                                with col2:
                                    if st.button(" 히스토리에 저장", type="primary", use_container_width=True, key="save_pie_multi"):
                                        add_to_full_history(**st.session_state.last_analysis)
                                        st.success(" 저장 완료!")
                                        st.balloons()
                                
                            except Exception as e:
                                st.error(f" 다중 범위 파이차트 생성 실패: {e}")
                                st.exception(e)
                                log_error("MultiRangePieChartError", "다중 범위 파이차트 오류", str(e))
                        
                        else:
                            st.error(" 분석할 수치 컬럼을 찾을 수 없습니다.")
                            st.info(f"사용 가능한 수치 컬럼: {', '.join(numeric_cols)}")
                    
                    # === 단일 범위 기반 파이차트 (기존) ===
                    elif range_based and threshold is not None:

fig.show()
"""
                                
                                # 히스토리 저장
                                add_to_full_history(
                                    question=user_question,
                                    result_type="다중범위_파이차트",
                                    figure=fig,
                                    data=range_counts,
                                    insights=insights_text,
                                    code=range_code,
                                    data_code=range_data_code,
                                    chart_type="파이차트",
                                    time_unit="N/A"
                                )
                                
                            except Exception as e:
                                st.error(f" 다중 범위 파이차트 생성 실패: {e}")
                                log_error("MultiRangePieChartError", "다중 범위 파이차트 오류", str(e))
                        
                        else:
                            st.error(" 분석할 수치 컬럼을 찾을 수 없습니다.")
                            st.info(f"사용 가능한 수치 컬럼: {', '.join(numeric_cols)}")
                    
                    # === 단일 범위 기반 파이차트 (기존) ===
                    elif range_based and threshold is not None:
                        st.info(f" 범위 기반 그룹핑 감지: **{threshold}** 기준")
                        
                        # 수치형 컬럼 찾기
                        value_col = None
                        if mentioned_col:
                            value_col = mentioned_col
                        elif numeric_cols:
                            for col in numeric_cols:
                                if any(kw in col.lower() for kw in ['wgt', 'unit', 'cnt', 'count', 'sum', 'total']):
                                    value_col = col
                                    break
                            if not value_col:
                                value_col = numeric_cols[0]
                            st.info(f" 분석 컬럼: **{value_col}**")
                        
                        if value_col:
                            try:
                                # 범위 기반 그룹 생성
                                df_copy = df_work.copy()
                                df_copy['range_group'] = df_copy[value_col].apply(
                                    lambda x: f'{threshold} 이하' if x <= threshold else f'{threshold} 초과'
                                )
                                
                                # 그룹별 개수 계산
                                range_counts = df_copy['range_group'].value_counts().reset_index()
                                range_counts.columns = ['범위', '개수']
                                
                                # 파이차트 생성
                                fig = px.pie(
                                    range_counts,
                                    names='범위',
                                    values='개수',
                                    title=f'{value_col} 값 기준 범위별 분포 (기준: {threshold})'
                                )
                                
                                fig.update_traces(textposition='inside', textinfo='percent+label+value')
                                fig.update_layout(height=500)
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # 상세 통계
                                with st.expander(" 상세 통계"):
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.markdown("**개수 및 비율:**")
                                        range_counts['비율(%)'] = (range_counts['개수'] / range_counts['개수'].sum() * 100).round(2)
                                        st.dataframe(range_counts, use_container_width=True)
                                    
                                    with col2:
                                        st.markdown("**실제 값 통계:**")
                                        group1 = df_copy[df_copy[value_col] <= threshold][value_col]
                                        group2 = df_copy[df_copy[value_col] > threshold][value_col]
                                        
                                        stats_df = pd.DataFrame({
                                            '범위': [f'{threshold} 이하', f'{threshold} 초과'],
                                            '평균': [group1.mean() if len(group1) > 0 else 0, 
                                                    group2.mean() if len(group2) > 0 else 0],
                                            '최소': [group1.min() if len(group1) > 0 else 0, 
                                                    group2.min() if len(group2) > 0 else 0],
                                            '최대': [group1.max() if len(group1) > 0 else 0, 
                                                    group2.max() if len(group2) > 0 else 0]
                                        })
                                        st.dataframe(stats_df, use_container_width=True)
                                
                                # 인사이트
                                total = range_counts['개수'].sum()
                                group1_cnt = range_counts[range_counts['범위'] == f'{threshold} 이하']['개수'].values[0] if f'{threshold} 이하' in range_counts['범위'].values else 0
                                group2_cnt = range_counts[range_counts['범위'] == f'{threshold} 초과']['개수'].values[0] if f'{threshold} 초과' in range_counts['범위'].values else 0
                                
                                pct1 = round(group1_cnt/total*100, 1)
                                pct2 = round(group2_cnt/total*100, 1)
                                
                                insights_text = f"""범위별 분포 인사이트:
- 전체: {total}개
- {threshold} 이하: {group1_cnt}개 ({pct1}%)
- {threshold} 초과: {group2_cnt}개 ({pct2}%)
"""
                                
                                if group1_cnt > group2_cnt:
                                    ratio1 = round(group1_cnt/group2_cnt, 1)
                                    insights_text += f"\n{threshold} 이하 구간이 더 많습니다 ({ratio1}배)"
                                elif group2_cnt > group1_cnt:
                                    ratio2 = round(group2_cnt/group1_cnt, 1)
                                    insights_text += f"\n{threshold} 초과 구간이 더 많습니다 ({ratio2}배)"
                                else:
                                    insights_text += f"\n두 구간이 비슷합니다"
                                
                                st.success(insights_text)
                                
                                # 코드 생성
                                range_data_code = f"""# 범위 기반 데이터 처리
import pandas as pd

# 1. 원본 데이터 로드
df = pd.read_csv('your_file.csv')
print(f"원본 데이터: {{len(df):,}}행")

# 2. 범위 기반 그룹 생성
df['range_group'] = df['{value_col}'].apply(
    lambda x: '{threshold} 이하' if x <= {threshold} else '{threshold} 초과'
)

# 3. 그룹별 개수 계산
range_counts = df['range_group'].value_counts().reset_index()
range_counts.columns = ['범위', '개수']

print(range_counts)
"""
                                
                                range_code = f"""# 범위 기반 파이차트 생성
import plotly.express as px

fig = px.pie(
    range_counts,
    names='범위',
    values='개수',
    title='{value_col} 값 기준 범위별 분포 (기준: {threshold})'
)

fig.update_traces(textposition='inside', textinfo='percent+label+value')
fig.update_layout(height=500)

fig.show()
"""
                                
                                # 히스토리 저장
                                add_to_full_history(
                                    question=user_question,
                                    result_type="범위별_파이차트",
                                    figure=fig,
                                    data=range_counts,
                                    insights=insights_text,
                                    code=range_code,
                                    data_code=range_data_code,
                                    chart_type="파이차트",
                                    time_unit="N/A"
                                )
                                
                            except Exception as e:
                                st.error(f" 범위 기반 파이차트 생성 실패: {e}")
                                log_error("RangePieChartError", "범위 파이차트 오류", str(e))
                        
                        else:
                            st.error(" 분석할 수치 컬럼을 찾을 수 없습니다.")
                            st.info(f"사용 가능한 수치 컬럼: {', '.join(numeric_cols)}")
                    
                    # === 일반 파이차트 (기존 로직) ===
                    else:
                        # 범주형 컬럼 찾기
                        cat_col = None
                        cat_cols = df_work.select_dtypes(include=['object']).columns.tolist()
                        
                        for col in cat_cols:
                            if col in user_question_lower:
                                cat_col = col
                                break
                        
                        # 컬럼 못 찾으면 첫 번째 범주형 컬럼 사용
                        if not cat_col and cat_cols:
                            cat_col = cat_cols[0]
                            st.info(f" 범주형 컬럼 자동 선택: **{cat_col}**")
                        
                        # === 핵심: 수치형 컬럼이 질문에 명시되었는지 확인 ===
                        value_col = None
                        use_count_based = True  # 기본은 개수 기반
                        
                        # 질문에 수치형 컬럼이 명시되어 있으면 값 기반
                        if mentioned_col:
                            value_col = mentioned_col
                            use_count_based = False
                            st.info(f" 값 기반 파이차트: **{value_col}** 합계 사용")
                        else:
                            st.info(f" 개수 기반 파이차트: **{cat_col}** 범주별 개수")
                        
                        if cat_col:
                            try:
                                # === 개수 기반 파이차트 ===
                                if use_count_based:
                                    # 범주별 개수 계산
                                    pie_data = df_work[cat_col].value_counts().reset_index()
                                    pie_data.columns = [cat_col, '개수']
                                    
                                    # 파이차트 생성
                                    fig = px.pie(
                                        pie_data,
                                        names=cat_col,
                                        values='개수',
                                        title=f'{cat_col}별 개수 비율'
                                    )
                                    
                                    fig.update_traces(textposition='inside', textinfo='percent+label+value')
                                    fig.update_layout(height=500)
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # 데이터 테이블
                                    with st.expander(" 데이터 테이블"):
                                        pie_data['비율(%)'] = (pie_data['개수'] / pie_data['개수'].sum() * 100).round(2)
                                        st.dataframe(pie_data, use_container_width=True)
                                    
                                    # 인사이트
                                    max_cat = pie_data.loc[pie_data['개수'].idxmax(), cat_col]
                                    max_val = pie_data['개수'].max()
                                    max_pct = (max_val / pie_data['개수'].sum() * 100)
                                    total_count = pie_data['개수'].sum()
                                    max_pct_val = round(max_pct, 1)
                                    
                                    insights_text = f"""
파이차트 인사이트 (개수 기반):
- 가장 많은 범주: {max_cat} ({max_val}개, {max_pct_val}%)
- 총 {len(pie_data)}개 범주
- 전체 개수: {total_count}개
                                    """
                                    
                                    st.success(insights_text)
                                    
                                    # 코드 생성
                                    pie_data_code = f"""# 개수 기반 데이터 처리
import pandas as pd

# 1. 원본 데이터 로드
df = pd.read_csv('your_file.csv')
print(f"원본 데이터: {{len(df):,}}행")

# 2. 범주별 개수 계산
pie_data = df['{cat_col}'].value_counts().reset_index()
pie_data.columns = ['{cat_col}', '개수']

print(f"처리된 데이터:")
print(pie_data)
"""
                                    
                                    pie_code = f"""# 파이차트 생성 (개수 기반)
import plotly.express as px

fig = px.pie(
    pie_data,
    names='{cat_col}',
    values='개수',
    title='{cat_col}별 개수 비율'
)

fig.update_traces(textposition='inside', textinfo='percent+label+value')
fig.update_layout(height=500)

fig.show()
"""
                                
                                # === 값 기반 파이차트 ===
                                else:
                                    # 범주별 합계 계산
                                    pie_data = df_work.groupby(cat_col)[value_col].sum().reset_index()
                                    pie_data.columns = [cat_col, f'{value_col}_합계']
                                    
                                    # 파이차트 생성
                                    fig = px.pie(
                                        pie_data,
                                        names=cat_col,
                                        values=f'{value_col}_합계',
                                        title=f'{cat_col}별 {value_col} 비율'
                                    )
                                    
                                    fig.update_traces(textposition='inside', textinfo='percent+label')
                                    fig.update_layout(height=500)
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # 데이터 테이블
                                    with st.expander(" 데이터 테이블"):
                                        pie_data['비율(%)'] = (pie_data[f'{value_col}_합계'] / pie_data[f'{value_col}_합계'].sum() * 100).round(2)
                                        st.dataframe(pie_data, use_container_width=True)
                                    
                                    # 인사이트
                                    max_cat = pie_data.loc[pie_data[f'{value_col}_합계'].idxmax(), cat_col]
                                    max_val = pie_data[f'{value_col}_합계'].max()
                                    max_pct = (max_val / pie_data[f'{value_col}_합계'].sum() * 100)
                                    total_sum = pie_data[f'{value_col}_합계'].sum()
                                    max_pct_val = round(max_pct, 1)
                                    max_val_round = round(max_val, 2)
                                    total_sum_round = round(total_sum, 2)
                                    
                                    insights_text = f"""
파이차트 인사이트 (값 기반):
- 가장 큰 비중: {max_cat} ({max_val_round}, {max_pct_val}%)
- 총 {len(pie_data)}개 범주
- 전체 합계: {total_sum_round}
                                    """
                                    
                                    st.success(insights_text)
                                    
                                    # 코드 생성
                                    pie_data_code = f"""# 값 기반 데이터 처리
import pandas as pd

# 1. 원본 데이터 로드
df = pd.read_csv('your_file.csv')
print(f"원본 데이터: {{len(df):,}}행")

# 2. 범주별 합계 계산
pie_data = df.groupby('{cat_col}')['{value_col}'].sum().reset_index()
pie_data.columns = ['{cat_col}', '{value_col}_합계']

print(f"처리된 데이터:")
print(pie_data)
"""
                                    
                                    pie_code = f"""# 파이차트 생성 (값 기반)
import plotly.express as px

fig = px.pie(
    pie_data,
    names='{cat_col}',
    values='{value_col}_합계',
    title='{cat_col}별 {value_col} 비율'
)

fig.update_traces(textposition='inside', textinfo='percent+label')
fig.update_layout(height=500)

fig.show()
"""
                                
                                # 히스토리 저장
                                add_to_full_history(
                                    question=user_question,
                                    result_type="파이차트_분석",
                                    figure=fig,
                                    data=pie_data,
                                    insights=insights_text,
                                    code=pie_code,
                                    data_code=pie_data_code,
                                    chart_type="파이차트",
                                    time_unit="N/A"
                                )
                                
                            except Exception as e:
                                st.error(f" 파이차트 생성 실패: {e}")
                                log_error("PieChartError", "파이차트 생성 오류", str(e))
                        
                        else:
                            st.error(" 파이차트에 필요한 범주형 컬럼을 찾을 수 없습니다.")
                            st.info(f"""
**파이차트 요구사항:**
- 범주형 컬럼: {', '.join(cat_cols) if cat_cols else '없음'}

 질문 예시:
- "md_shft 파이차트" -> 개수 기반
- "md_shft별 prod_wgt 파이차트" -> 값 기반
                            """)
                            log_error("PieChartError", "필요 컬럼 없음", f"범주: {cat_cols}")
                
                # === 우선순위 2: 간단한 통계 ===
                elif "행" in user_question or "row" in user_question_lower:
                    result = f" 데이터 행 수: **{len(df_work):,}개**"
                    st.success(result)
                    add_to_full_history(user_question, "통계", insights=result, chart_type="N/A", time_unit="N/A")
                
                elif "컬럼" in user_question and not wants_graph:
                    result = f" 컬럼: {', '.join(df_work.columns.tolist())}"
                    st.success(result)
                    add_to_full_history(user_question, "통계", insights=result, chart_type="N/A", time_unit="N/A")
                
                elif "평균" in user_question and mentioned_col and not wants_graph and not is_time_series:
                    avg = df_work[mentioned_col].mean()
                    result = f" {mentioned_col} 평균: **{avg:,.2f}**"
                    st.success(result)
                    add_to_full_history(user_question, "통계", insights=result, chart_type="N/A", time_unit="N/A")
                
                elif "결측치" in user_question:
                    null_cols = df_work.isnull().sum()
                    null_cols = null_cols[null_cols > 0]
                    if len(null_cols) > 0:
                        st.write("**결측치가 있는 컬럼:**")
                        st.dataframe(null_cols)
                        add_to_full_history(user_question, "결측치", data=pd.DataFrame(null_cols), chart_type="N/A", time_unit="N/A")
                    else:
                        result = " 결측치가 없습니다!"
                        st.success(result)
                        add_to_full_history(user_question, "결측치", insights=result, chart_type="N/A", time_unit="N/A")
                
                else:
                    st.warning("⚠️ 질문을 이해하지 못했습니다.")
                    log_error("QuestionParseError", "질문 파싱 실패", user_question)
                    st.info("""
** 질문 예시:**

**시간 단위:**
- "md_shft별로 prod_wgt **일별** 추이 그래프"
- "prod_wgt **월별** 추이"

**그래프 타입:**
- "prod_wgt 월별 **막대그래프**"
- "md_shft별 **파이차트**"

**범위 지정:**
- "prod_wgt **1월부터 8월까지** 월별 추이"
- "wat_unit **3월~7월** 막대그래프"
- "prod_wgt **2024-01-01부터 2024-06-30까지** 일별 추이"

**기타:**
- "prod_wgt 평균은?"
                    """)
                
                if 'sample_question' in st.session_state:
                    del st.session_state.sample_question
            
            except Exception as e:
                log_error("UnexpectedError", "예상치 못한 오류", traceback.format_exc())
                st.error(f" 오류 발생: {e}")
                st.error("상세 오류는 하단 ' 에러 로그' 섹션을 확인하세요.")
        
        else:
            st.warning("⚠️ 질문을 입력하세요")
    
    # --- 분석 히스토리 ---
    if len(st.session_state.analysis_history) > 0:
        st.divider()
        st.subheader(" 분석 히스토리")
        
        st.write(f"**총 {len(st.session_state.analysis_history)}개의 분석 결과**")
        
        for idx, entry in enumerate(reversed(st.session_state.analysis_history), 1):
            with st.expander(f"**{idx}. [{entry['timestamp']}]** {entry['question']}", expanded=(idx == 1)):
                st.write(f"**분석 유형:** {entry['result_type']}")
                
                if entry['figure'] is not None:
                    # 고유 key로 히스토리 차트 표시
                    history_key = f"history_{entry['id']}_{idx}"
                    st.plotly_chart(entry['figure'], use_container_width=True, key=history_key)
                    
                    # 그래프 생성 코드 표시
                    with st.expander(" 그래프 생성 코드", expanded=False):
                        if entry.get('code'):
                            st.code(entry['code'], language="python")
                        else:
                            st.info("코드 정보가 저장되지 않았습니다.")
                
                if entry['data'] is not None:
                    st.write("**데이터:**")
                    st.dataframe(pd.DataFrame(entry['data']))
                    
                    # 데이터 처리 코드 표시
                    with st.expander(" 데이터 처리 코드", expanded=False):
                        if entry.get('data_code'):
                            st.code(entry['data_code'], language="python")
                        else:
                            st.info("데이터 처리 코드가 저장되지 않았습니다.")
                
                if entry['insights']:
                    st.write("**인사이트:**")
                    st.markdown(entry['insights'])
        
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("️ 히스토리 초기화", key="clear_history"):
                st.session_state.analysis_history = []
                st.rerun()

# --- 에러 로그 섹션 ---
if len(st.session_state.error_logs) > 0:
    st.divider()
    st.subheader(" 에러 로그")
    
    with st.expander(f"⚠️ 에러 {len(st.session_state.error_logs)}개 발생 - 클릭하여 확인", expanded=False):
        for idx, error in enumerate(reversed(st.session_state.error_logs[-10:]), 1):
            st.error(f"""
**에러 {idx}**
- 시간: {error['timestamp']}
- 타입: {error['type']}
- 메시지: {error['message']}
            """)
            
            with st.expander(f"상세 정보 {idx}"):
                st.code(error['details'], language="python")
        
        if st.button("️ 에러 로그 초기화", key="clear_errors"):
            st.session_state.error_logs = []
            st.rerun()

st.divider()
st.caption(" 철강 설비 AI 대시보드 v13.0 | 최적화된 히스토리 UI | Gemini 2.5")

# === Google Sheets 히스토리 (그래프는 항상, 상세정보는 토글) ===
render_full_history_ui()
