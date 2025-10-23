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

# 경고 메시지 무시
warnings.filterwarnings('ignore')

# --- 페이지 설정 ---
st.set_page_config(page_title="철강 설비 AI 대시보드", layout="wide")
st.title("🤖 철강 설비 AI 에이전트 대시보드")

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
def add_to_history(question, result_type, figure=None, data=None, insights=None, code=None):
    """분석 결과를 히스토리에 추가"""
    history_entry = {
        "id": str(uuid.uuid4()),  # 고유 ID 생성
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "result_type": result_type,
        "figure": figure,
        "data": data.to_dict() if data is not None else None,
        "insights": insights,
        "code": code
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
        st.error(f"❌ {error['message']}")
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
            print(f"✅ 성공: {model_name}")
            st.success(f"✅ Google Gemini ({model_name}) 로드 완료!")
            return llm
            
        except Exception as e:
            log_error("ModelLoadError", f"{model_name} 로드 실패", str(e))
            print(f"❌ 실패: {model_name} - {e}")
            continue
    
    st.warning("⚠️ Gemini 로드 실패: AI 인사이트 없이 기본 분석 모드로 사용 가능합니다.")
    return None

llm = get_llm()

# --- 데이터 로드 ---
st.divider()
st.subheader("📊 설비 데이터")

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
        st.success(f"✅ 로드 완료: {len(df):,}행 × {len(df.columns)}컬럼")
        return df
    except Exception as e:
        log_error("DataLoadError", "CSV 로딩 실패", str(e))
        st.error(f"❌ 로딩 오류: {e}")
        return None

df_facility = None

if uploaded_file:
    df_facility = load_data(uploaded_file)
else:
    st.info("💡 CSV 파일을 업로드하세요")
    
    if st.button("🎲 샘플 데이터로 테스트"):
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
        st.success("✅ 샘플 데이터 생성 (300일, 이상치 포함)!")
        st.rerun()

if df_facility is not None:
    with st.expander("🔍 데이터 미리보기"):
        st.dataframe(df_facility.head(10))
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**📋 기본 정보:**")
        st.write(f"- 행 수: {len(df_facility):,}")
        st.write(f"- 컬럼 수: {len(df_facility.columns)}")
        st.write(f"- 메모리: {df_facility.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    with col2:
        numeric_cols = df_facility.select_dtypes(include=['int64', 'float64']).columns.tolist()
        st.write(f"**📊 데이터 타입:**")
        st.write(f"- 수치형: {len(numeric_cols)}개")
        st.write(f"- 범주형: {len(df_facility.select_dtypes(include=['object']).columns)}개")
    
    with st.expander("📋 컬럼 상세 정보"):
        col_info = pd.DataFrame({
            '컬럼명': df_facility.columns,
            '데이터타입': df_facility.dtypes.values,
            '결측치': df_facility.isnull().sum().values,
            '고유값수': [df_facility[col].nunique() for col in df_facility.columns]
        })
        st.dataframe(col_info)
    
    if numeric_cols:
        with st.expander("📊 수치형 컬럼 통계"):
            st.dataframe(df_facility[numeric_cols].describe().T)
    
    # --- 데이터 전처리 옵션 ---
    st.divider()
    st.subheader("🔧 데이터 전처리 설정")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        use_outlier_removal = st.checkbox("🎯 이상치 제거 사용", value=False, 
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
    st.subheader("💬 AI에게 질문하기")
    
    sample_qs = {
        "Q1": "prod_wgt 평균은?",
        "Q2": "prod_wgt 일별 추이",
        "Q3": "md_shft별 prod_wgt 일별 추이",
        "Q4": "prod_wgt 월별 추이",
        "Q5": "md_shft별 prod_wgt 월별 추이"
    }
    
    st.write("**💡 샘플 질문:**")
    cols = st.columns(5)
    for idx, (key, q) in enumerate(sample_qs.items()):
        if cols[idx].button(key, help=q, key=f"sample_q_{idx}"):
            st.session_state.sample_question = q
    
    user_question = st.text_input(
        "질문:",
        value=st.session_state.get('sample_question', ''),
        placeholder="예: md_shft별로 prod_wgt 일별 평균 추이 그래프 (또는 월별)"
    )
    
    if st.button("🚀 분석", type="primary"):
        if user_question:
            try:
                # === 질문 분석 (개선된 로직) ===
                user_question_lower = user_question.lower()
                
                graph_keywords = ["그래프", "선그래프", "막대그래프", "차트", "추이", "변화", "표현", "그려", "시각화"]
                
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
                st.info(f"🔍 감지된 시간 단위: **{time_unit_kr}** ({detected_reason})")
                
                # 날짜 컬럼 찾기
                date_col = None
                for col in df_facility.columns:
                    if 'date' in col.lower() and pd.api.types.is_datetime64_any_dtype(df_facility[col]):
                        date_col = col
                        break
                
                if not date_col and (wants_graph or is_time_series):
                    st.error("❌ 날짜 컬럼을 찾을 수 없습니다. 컬럼명에 'date'가 포함되어야 합니다.")
                    log_error("ColumnNotFound", "날짜 컬럼 없음", f"사용 가능 컬럼: {df_facility.columns.tolist()}")
                
                # 분석 컬럼 찾기
                mentioned_col = None
                for col in numeric_cols:
                    if col in user_question_lower:
                        mentioned_col = col
                        break
                
                if not mentioned_col and (wants_graph or is_time_series):
                    st.error(f"❌ 분석할 수치 컬럼을 찾을 수 없습니다. 질문에 컬럼명을 포함해주세요: {', '.join(numeric_cols)}")
                    log_error("ColumnNotFound", "수치 컬럼 없음", f"질문: {user_question}")
                
                # 그룹 컬럼 찾기
                group_col = None
                if is_multi_series:
                    cat_cols = df_facility.select_dtypes(include=['object']).columns
                    for col in cat_cols:
                        if col in user_question_lower:
                            group_col = col
                            break
                
                # === 우선순위 1: 그래프 요청 ===
                if (wants_graph or is_time_series) and date_col and mentioned_col:
                    
                    # 데이터 복사
                    temp_df = df_facility.copy()
                    
                    # 이상치 제거
                    outlier_info = None
                    removed_count = 0
                    if use_outlier_removal:
                        st.info(f"🔧 이상치 제거 중... (방법: {outlier_method}, 임계값: {outlier_threshold})")
                        temp_df, removed_count, outlier_info = remove_outliers(
                            temp_df, mentioned_col, outlier_method, outlier_threshold
                        )
                        
                        if removed_count > 0:
                            st.success(f"✅ 이상치 제거 완료: {removed_count:,}개 행 제거 ({removed_count/len(df_facility)*100:.1f}%)")
                        else:
                            st.info("ℹ️ 제거된 이상치가 없습니다.")
                    
                    # ⭐ 시간 단위에 따라 그룹화
                    if time_unit == "day":
                        temp_df['time_group'] = temp_df[date_col].dt.date
                        x_label = "날짜"
                    else:  # month
                        temp_df['time_group'] = temp_df[date_col].dt.month
                        x_label = "월"
                    
                    # 기간 필터링
                    if "1월" in user_question and "10월" in user_question and time_unit == "month":
                        temp_df = temp_df[temp_df['time_group'].between(1, 10)]
                    
                    if is_multi_series and group_col:
                        # === 다중 계열 분석 ===
                        st.markdown(f"### 📈 계열별 {time_unit_kr} 추이 그래프")
                        
                        if use_outlier_removal and removed_count > 0:
                            st.caption(f"💡 이상치 제거 적용됨: {removed_count:,}개 데이터 포인트 제거")
                        
                        multi = temp_df.groupby(['time_group', group_col])[mentioned_col].mean().reset_index()
                        multi.columns = [x_label, group_col, mentioned_col]
                        
                        # 고유 key로 차트 생성
                        chart_key = f"chart_{uuid.uuid4().hex[:8]}"
                        fig = px.line(multi, x=x_label, y=mentioned_col, color=group_col,
                                    markers=True, 
                                    title=f'{mentioned_col}의 {group_col}별 {time_unit_kr} 평균 추이{"(이상치 제거)" if use_outlier_removal else ""}')
                        fig.update_xaxes(title=x_label)
                        fig.update_yaxes(title=f"{mentioned_col} 평균")
                        fig.update_layout(legend_title=group_col, height=500)
                        st.plotly_chart(fig, use_container_width=True, key=chart_key)
                        
                        with st.expander("📊 계열별 데이터 테이블"):
                            pivot = multi.pivot(index=x_label, columns=group_col, values=mentioned_col)
                            st.dataframe(pivot)
                        
                        with st.expander("💻 그래프 생성 코드"):
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
                        st.markdown("### 🎯 계열별 핵심 인사이트")
                        
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
                                    st.success(f"**🤖 AI 인사이트:**\n\n{ai_insight}")
                                    insights_text += f"\n🤖 AI 분석:\n{ai_insight}"
                                except Exception as e:
                                    log_error("AIInsightError", "AI 인사이트 생성 실패", str(e))
                                    st.warning(f"⚠️ AI 인사이트 생성 실패: {e}")
                        
                        # 히스토리에 추가
                        multi_code = f"""import plotly.express as px
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
                        
                        add_to_history(
                            question=user_question,
                            result_type=f"계열별_{time_unit_kr}_추이",
                            figure=fig,
                            data=multi,
                            insights=insights_text,
                            code=multi_code
                        )
                    
                    else:
                        # === 단일 계열 분석 ===
                        st.markdown(f"### 📈 {time_unit_kr} 추이 그래프")
                        
                        if use_outlier_removal and removed_count > 0:
                            st.caption(f"💡 이상치 제거 적용됨: {removed_count:,}개 데이터 포인트 제거")
                        
                        time_data = temp_df.groupby('time_group')[mentioned_col].mean().reset_index()
                        time_data.columns = [x_label, mentioned_col]
                        
                        # 고유 key로 차트 생성
                        chart_key = f"chart_{uuid.uuid4().hex[:8]}"
                        fig = px.line(time_data, x=x_label, y=mentioned_col,
                                    markers=True, 
                                    title=f'{mentioned_col}의 {time_unit_kr} 평균 추이{"(이상치 제거)" if use_outlier_removal else ""}')
                        fig.update_xaxes(title=x_label)
                        fig.update_yaxes(title=f"{mentioned_col} 평균")
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True, key=chart_key)
                        
                        with st.expander("📊 데이터 테이블"):
                            st.dataframe(time_data)
                        
                        with st.expander("💻 그래프 생성 코드"):
                            code = f"""import plotly.express as px
import pandas as pd

# 데이터 준비
data = {time_data.to_dict('list')}
df = pd.DataFrame(data)

# 그래프 생성
fig = px.line(df, 
              x='{x_label}', 
              y='{mentioned_col}',
              markers=True, 
              title='{mentioned_col}의 {time_unit_kr} 평균 추이{"(이상치 제거)" if use_outlier_removal else ""}')

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
**🎯 핵심 인사이트:**
- 최고점: {max_time} ({max_val:,.2f})
- 최저점: {min_time} ({min_val:,.2f})
- 평균: {avg_val:,.2f}
- 변동폭: {max_val - min_val:,.2f} ({((max_val/min_val - 1) * 100):.1f}% 증가)
                        """
                        st.info(insights_text)
                        
                        # 히스토리에 추가
                        single_code = f"""import plotly.express as px
import pandas as pd

# 데이터 준비
data = {time_data.to_dict('list')}
df = pd.DataFrame(data)

# 그래프 생성
fig = px.line(df, 
              x='{x_label}', 
              y='{mentioned_col}',
              markers=True, 
              title='{mentioned_col}의 {time_unit_kr} 평균 추이{"(이상치 제거)" if use_outlier_removal else ""}')

fig.update_xaxes(title='{x_label}')
fig.update_yaxes(title='{mentioned_col} 평균')
fig.update_layout(height=500)

fig.show()"""
                        
                        add_to_history(
                            question=user_question,
                            result_type=f"{time_unit_kr}_추이",
                            figure=fig,
                            data=time_data,
                            insights=insights_text,
                            code=single_code
                        )
                
                # === 우선순위 2: 간단한 통계 ===
                elif "행" in user_question or "row" in user_question_lower:
                    result = f"📊 데이터 행 수: **{len(df_facility):,}개**"
                    st.success(result)
                    add_to_history(user_question, "통계", insights=result)
                
                elif "컬럼" in user_question and not wants_graph:
                    result = f"📋 컬럼: {', '.join(df_facility.columns.tolist())}"
                    st.success(result)
                    add_to_history(user_question, "통계", insights=result)
                
                elif "평균" in user_question and mentioned_col and not wants_graph and not is_time_series:
                    avg = df_facility[mentioned_col].mean()
                    result = f"📊 {mentioned_col} 평균: **{avg:,.2f}**"
                    st.success(result)
                    add_to_history(user_question, "통계", insights=result)
                
                elif "결측치" in user_question:
                    null_cols = df_facility.isnull().sum()
                    null_cols = null_cols[null_cols > 0]
                    if len(null_cols) > 0:
                        st.write("**결측치가 있는 컬럼:**")
                        st.dataframe(null_cols)
                        add_to_history(user_question, "결측치", data=pd.DataFrame(null_cols))
                    else:
                        result = "✅ 결측치가 없습니다!"
                        st.success(result)
                        add_to_history(user_question, "결측치", insights=result)
                
                else:
                    st.warning("⚠️ 질문을 이해하지 못했습니다.")
                    log_error("QuestionParseError", "질문 파싱 실패", user_question)
                    st.info("""
**💡 질문 예시:**
- "md_shft별로 prod_wgt **일별** 추이 그래프"
- "prod_wgt **월별** 추이"
- "prod_wgt 평균은?"
                    """)
                
                if 'sample_question' in st.session_state:
                    del st.session_state.sample_question
            
            except Exception as e:
                log_error("UnexpectedError", "예상치 못한 오류", traceback.format_exc())
                st.error(f"❌ 오류 발생: {e}")
                st.error("상세 오류는 하단 '🐛 에러 로그' 섹션을 확인하세요.")
        
        else:
            st.warning("⚠️ 질문을 입력하세요")
    
    # --- 분석 히스토리 ---
    if len(st.session_state.analysis_history) > 0:
        st.divider()
        st.subheader("📚 분석 히스토리")
        
        st.write(f"**총 {len(st.session_state.analysis_history)}개의 분석 결과**")
        
        for idx, entry in enumerate(reversed(st.session_state.analysis_history), 1):
            with st.expander(f"**{idx}. [{entry['timestamp']}]** {entry['question']}", expanded=(idx == 1)):
                st.write(f"**분석 유형:** {entry['result_type']}")
                
                if entry['figure'] is not None:
                    # 고유 key로 히스토리 차트 표시
                    history_key = f"history_{entry['id']}_{idx}"
                    st.plotly_chart(entry['figure'], use_container_width=True, key=history_key)
                    
                    # 그래프 생성 코드 표시
                    with st.expander("💻 그래프 생성 코드", expanded=False):
                        if entry.get('code'):
                            st.code(entry['code'], language="python")
                        else:
                            st.info("코드 정보가 저장되지 않았습니다.")
                
                if entry['data'] is not None:
                    st.write("**데이터:**")
                    st.dataframe(pd.DataFrame(entry['data']))
                
                if entry['insights']:
                    st.write("**인사이트:**")
                    st.markdown(entry['insights'])
        
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("🗑️ 히스토리 초기화", key="clear_history"):
                st.session_state.analysis_history = []
                st.rerun()

# --- 에러 로그 섹션 ---
if len(st.session_state.error_logs) > 0:
    st.divider()
    st.subheader("🐛 에러 로그")
    
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
        
        if st.button("🗑️ 에러 로그 초기화", key="clear_errors"):
            st.session_state.error_logs = []
            st.rerun()

st.divider()
st.caption("🔧 철강 설비 AI 대시보드 v10.2 | 키워드 감지 + 중복 ID 해결 + 코드 표시 | Gemini 2.5")
