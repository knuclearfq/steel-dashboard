import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
import warnings
import json
import re
import traceback
from datetime import datetime

# 경고 메시지 무시
warnings.filterwarnings('ignore')

# --- 페이지 설정 ---
st.set_page_config(page_title="철강 설비 AI 대시보드", layout="wide")
st.title("🤖 철강 설비 AI 에이전트 대시보드")

# 세션 상태 초기화
if 'error_logs' not in st.session_state:
    st.session_state.error_logs = []

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

# --- 이상치 제거 함수 ---
def remove_outliers(df, column, method='iqr', threshold=1.5):
    """
    이상치 제거 함수
    
    Parameters:
    - df: DataFrame
    - column: 대상 컬럼명
    - method: 'iqr' (사분위수), 'zscore' (Z-점수), 'percentile' (백분위수)
    - threshold: 임계값 (IQR: 1.5배, Z-score: 3, Percentile: 상하위 1%)
    
    Returns:
    - 정제된 DataFrame, 제거된 행 수
    """
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
        st.error("❌ Google Gemini API 키가 설정되지 않았습니다.")
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
        
        dates = pd.date_range('2024-01-01', periods=2631, freq='4H')
        
        # 7월에 이상치 포함한 샘플 데이터
        wat_unit_values = []
        for date in dates:
            month = date.month
            if month == 7:
                # 7월에 일부 이상치 추가
                if np.random.random() > 0.95:  # 5% 확률로 이상치
                    wat_unit_values.append(np.random.uniform(900, 1100))
                else:
                    wat_unit_values.append(np.random.normal(630, 30))
            else:
                wat_unit_values.append(np.random.normal(450, 50))
        
        df_facility = pd.DataFrame({
            'wrk_date': dates,
            'md_shft': np.random.choice(['A', 'B', 'C'], 2631),
            'prod_wgt': np.random.normal(1500, 300, 2631),
            'eaf_wat_sum': np.random.normal(5000, 1000, 2631),
            'wat_unit': wat_unit_values
        })
        st.success("✅ 샘플 데이터 생성 (이상치 포함)!")
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
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    sample_qs = {
        "Q1": "prod_wgt 평균은?",
        "Q2": "wat_unit 월별 추이",
        "Q3": "md_shft별 wat_unit 월별 추이",
        "Q4": "데이터 행 수는?",
        "Q5": "결측치가 있는 컬럼은?"
    }
    
    st.write("**💡 샘플 질문:**")
    cols = st.columns(5)
    for idx, (key, q) in enumerate(sample_qs.items()):
        if cols[idx].button(key, help=q):
            st.session_state.sample_question = q
    
    user_question = st.text_input(
        "질문:",
        value=st.session_state.get('sample_question', ''),
        placeholder="예: md_shft별로 wat_unit 월별 추이를 선그래프로 (이상치 제거 체크박스 활성화하면 자동 적용)"
    )
    
    if st.button("🚀 분석", type="primary"):
        if user_question:
            try:
                # === 질문 분석 ===
                graph_keywords = ["그래프", "선그래프", "막대그래프", "차트", "추이", "변화", "표현", "그려", "시각화"]
                time_keywords = ["월별", "일별", "주별", "년별", "기간별", "시계열"]
                multi_keywords = ["계열", "조로", "구분", "별로", "분리", "그룹별", "나누어", "각각"]
                
                wants_graph = any(kw in user_question for kw in graph_keywords)
                is_time_series = any(kw in user_question for kw in time_keywords)
                is_multi_series = any(kw in user_question for kw in multi_keywords)
                
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
                    if col in user_question.lower():
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
                        if col in user_question.lower():
                            group_col = col
                            break
                
                # === 우선순위 1: 그래프 요청 ===
                if (wants_graph or is_time_series) and date_col and mentioned_col:
                    
                    # 데이터 복사
                    temp_df = df_facility.copy()
                    
                    # 이상치 제거 (옵션 활성화 시)
                    outlier_info = None
                    if use_outlier_removal:
                        st.info(f"🔧 이상치 제거 중... (방법: {outlier_method}, 임계값: {outlier_threshold})")
                        temp_df, removed_count, outlier_info = remove_outliers(
                            temp_df, mentioned_col, outlier_method, outlier_threshold
                        )
                        
                        if removed_count > 0:
                            st.success(f"✅ 이상치 제거 완료: {removed_count:,}개 행 제거 ({removed_count/len(df_facility)*100:.1f}%)")
                        else:
                            st.info("ℹ️ 제거된 이상치가 없습니다.")
                    
                    temp_df['month'] = temp_df[date_col].dt.month
                    
                    # 1-10월 필터링
                    if "1월" in user_question and "10월" in user_question:
                        temp_df = temp_df[temp_df['month'].between(1, 10)]
                    
                    if is_multi_series and group_col:
                        # === 다중 계열 분석 ===
                        st.markdown("### 📈 계열별 월별 추이 그래프")
                        
                        if use_outlier_removal and removed_count > 0:
                            st.caption(f"💡 이상치 제거 적용됨: {removed_count:,}개 데이터 포인트 제거")
                        
                        multi = temp_df.groupby(['month', group_col])[mentioned_col].mean().reset_index()
                        multi.columns = ['월', group_col, mentioned_col]
                        
                        fig = px.line(multi, x='월', y=mentioned_col, color=group_col,
                                    markers=True, 
                                    title=f'{mentioned_col}의 {group_col}별 월별 추이{"(이상치 제거)" if use_outlier_removal else ""}')
                        fig.update_xaxes(title="월", dtick=1)
                        fig.update_yaxes(title=f"{mentioned_col} 평균")
                        fig.update_layout(legend_title=group_col, height=500)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        with st.expander("📊 계열별 데이터 테이블"):
                            pivot = multi.pivot(index='월', columns=group_col, values=mentioned_col)
                            st.dataframe(pivot)
                        
                        # 프로세스 표시
                        process = {
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "question_type": "계열별 월별 추이",
                            "outlier_removal": use_outlier_removal,
                            "outlier_info": outlier_info,
                            "removed_count": removed_count if use_outlier_removal else 0,
                            "date_column": date_col,
                            "value_column": mentioned_col,
                            "group_column": group_col,
                            "groups": temp_df[group_col].unique().tolist(),
                            "steps": [
                                f"✅ 질문 유형: 계열별 월별 추이 그래프",
                                f"✅ 날짜 컬럼: {date_col}",
                                f"✅ 분석 컬럼: {mentioned_col}",
                                f"✅ 그룹 컬럼: {group_col}",
                            ]
                        }
                        
                        if use_outlier_removal:
                            process["steps"].extend([
                                f"✅ 이상치 제거: {outlier_method} 방법 ({removed_count}개 제거)",
                                f"   - {outlier_info['method']}",
                                f"   - 임계값: {outlier_threshold}"
                            ])
                        
                        process["steps"].extend([
                            f"✅ 월 추출: df['{date_col}'].dt.month",
                            f"✅ 그룹화: groupby(['month', '{group_col}'])['{mentioned_col}'].mean()",
                            "✅ 다중 선그래프 생성 완료"
                        ])
                        
                        process["pandas_code"] = f"df.groupby([df['{date_col}'].dt.month, '{group_col}'])['{mentioned_col}'].mean()"
                        process["plotly_code"] = f"px.line(data, x='월', y='{mentioned_col}', color='{group_col}', markers=True)"
                        
                        with st.expander("🔍 분석 프로세스"):
                            st.write("**📋 실행 단계:**")
                            for step in process["steps"]:
                                st.write(step)
                            
                            if use_outlier_removal and outlier_info:
                                st.write("\n**🎯 이상치 제거 상세:**")
                                st.json(outlier_info)
                            
                            st.write("\n**💻 Pandas 코드:**")
                            st.code(process["pandas_code"], language="python")
                            
                            st.write("\n**💻 Plotly 코드:**")
                            st.code(process["plotly_code"], language="python")
                            
                            st.write("\n**🔧 전체 프로세스:**")
                            st.json(process)
                        
                        # 계열별 인사이트
                        st.markdown("### 🎯 계열별 핵심 인사이트")
                        
                        for group in sorted(temp_df[group_col].unique()):
                            group_data = multi[multi[group_col] == group]
                            if len(group_data) > 0:
                                max_month = group_data.loc[group_data[mentioned_col].idxmax(), '월']
                                max_value = group_data[mentioned_col].max()
                                min_month = group_data.loc[group_data[mentioned_col].idxmin(), '월']
                                min_value = group_data[mentioned_col].min()
                                avg_value = group_data[mentioned_col].mean()
                                
                                st.info(f"""
                                **{group_col} = {group}**
                                - 최고점: {int(max_month)}월 ({max_value:,.2f})
                                - 최저점: {int(min_month)}월 ({min_value:,.2f})
                                - 평균: {avg_value:,.2f}
                                - 변동폭: {max_value - min_value:,.2f} ({((max_value/min_value - 1) * 100):.1f}% 증가)
                                """)
                        
                        # AI 인사이트
                        if llm:
                            with st.spinner("AI 인사이트 생성 중..."):
                                try:
                                    prompt = f"""
다음은 {group_col}별 {mentioned_col}의 월별 평균 데이터입니다{"(이상치 제거 후)" if use_outlier_removal else ""}:
{multi.to_string()}

철강 설비 데이터 전문가로서, 각 그룹별로 주목할 만한 특징과 차이점을 한국어로 3-4가지 설명하세요.
"""
                                    insight = llm.invoke(prompt)
                                    st.success(f"**🤖 AI 인사이트:**\n\n{insight.content}")
                                except Exception as e:
                                    log_error("AIInsightError", "AI 인사이트 생성 실패", str(e))
                                    st.warning(f"⚠️ AI 인사이트 생성 실패: {e}")
                    
                    else:
                        # === 단일 계열 분석 ===
                        st.markdown("### 📈 월별 추이 그래프")
                        
                        if use_outlier_removal and removed_count > 0:
                            st.caption(f"💡 이상치 제거 적용됨: {removed_count:,}개 데이터 포인트 제거")
                        
                        monthly = temp_df.groupby('month')[mentioned_col].mean().reset_index()
                        monthly.columns = ['월', mentioned_col]
                        
                        fig = px.line(monthly, x='월', y=mentioned_col,
                                    markers=True, 
                                    title=f'{mentioned_col}의 월별 평균 추이{"(이상치 제거)" if use_outlier_removal else ""}')
                        fig.update_xaxes(title="월", dtick=1)
                        fig.update_yaxes(title=f"{mentioned_col} 평균")
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        with st.expander("📊 월별 데이터 테이블"):
                            st.dataframe(monthly)
                        
                        # 인사이트
                        max_month = monthly.loc[monthly[mentioned_col].idxmax(), '월']
                        max_val = monthly[mentioned_col].max()
                        min_month = monthly.loc[monthly[mentioned_col].idxmin(), '월']
                        min_val = monthly[mentioned_col].min()
                        avg_val = monthly[mentioned_col].mean()
                        
                        st.info(f"""
**🎯 핵심 인사이트:**
- 최고점: {int(max_month)}월 ({max_val:,.2f})
- 최저점: {int(min_month)}월 ({min_val:,.2f})
- 평균: {avg_val:,.2f}
- 변동폭: {max_val - min_val:,.2f} ({((max_val/min_val - 1) * 100):.1f}% 증가)
                        """)
                
                # === 우선순위 2: 간단한 통계 ===
                elif "행" in user_question or "row" in user_question.lower():
                    st.success(f"📊 데이터 행 수: **{len(df_facility):,}개**")
                
                elif "컬럼" in user_question and not wants_graph:
                    st.success(f"📋 컬럼: {', '.join(df_facility.columns.tolist())}")
                
                elif "평균" in user_question and mentioned_col and not wants_graph and not is_time_series:
                    avg = df_facility[mentioned_col].mean()
                    st.success(f"📊 {mentioned_col} 평균: **{avg:,.2f}**")
                
                elif "결측치" in user_question:
                    null_cols = df_facility.isnull().sum()
                    null_cols = null_cols[null_cols > 0]
                    if len(null_cols) > 0:
                        st.write("**결측치가 있는 컬럼:**")
                        st.dataframe(null_cols)
                    else:
                        st.success("✅ 결측치가 없습니다!")
                
                else:
                    st.warning("⚠️ 질문을 이해하지 못했습니다.")
                    log_error("QuestionParseError", "질문 파싱 실패", user_question)
                    st.info("""
**💡 질문 예시:**
- "md_shft별로 wat_unit 월별 추이를 선그래프로"
- "wat_unit의 월별 추이 그래프"
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
    
    # --- 수동 그래프 ---
    st.divider()
    st.subheader("📈 수동 그래프 생성")
    
    col1, col2 = st.columns(2)
    with col1:
        chart_type = st.selectbox("차트 타입", ["선그래프", "막대", "히스토그램", "박스플롯"])
    with col2:
        if numeric_cols:
            selected_col = st.selectbox("분석할 컬럼", numeric_cols)
    
    if st.button("📊 그래프 생성", type="secondary"):
        try:
            if selected_col:
                if chart_type == "선그래프":
                    fig = px.line(df_facility.head(100), y=selected_col, title=f"{selected_col} 추이 (최근 100개)")
                elif chart_type == "막대":
                    fig = px.bar(df_facility.head(50), y=selected_col, title=f"{selected_col} (최근 50개)")
                elif chart_type == "히스토그램":
                    fig = px.histogram(df_facility, x=selected_col, title=f"{selected_col} 분포")
                else:
                    fig = px.box(df_facility, y=selected_col, title=f"{selected_col} 박스플롯")
                
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            log_error("ManualGraphError", "수동 그래프 생성 실패", str(e))
            st.error(f"❌ 그래프 생성 오류: {e}")

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
        
        if st.button("🗑️ 에러 로그 초기화"):
            st.session_state.error_logs = []
            st.rerun()

st.divider()
st.caption("🔧 철강 설비 AI 대시보드 v9.0 | 이상치 제거 + 에러 표시 | Gemini 2.5")
