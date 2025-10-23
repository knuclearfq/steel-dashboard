import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
import warnings
import json
import re
from datetime import datetime

# 경고 메시지 무시
warnings.filterwarnings('ignore')

# --- 페이지 설정 ---
st.set_page_config(page_title="철강 설비 AI 대시보드", layout="wide")
st.title("🤖 철강 설비 AI 에이전트 대시보드")

# --- Gemini LLM 로드 (여러 모델 시도) ---
@st.cache_resource
def get_llm():
    """Google Gemini LLM 객체를 로드하는 함수 (여러 모델 시도)"""
    print("Gemini LLM 로드 시도...")
    
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        st.error("""
        ❌ langchain-google-genai 패키지가 설치되지 않았습니다.
        
        requirements.txt에 다음 추가:
        ```
        langchain-google-genai>=1.0.0
        ```
        """)
        return None
    
    # API 키 가져오기
    api_key = st.secrets.get("GOOGLE_API_KEY") if "GOOGLE_API_KEY" in st.secrets else os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        st.error("""
        ❌ Google Gemini API 키가 설정되지 않았습니다.
        
        **API 키 발급 (무료):**
        1. https://aistudio.google.com/app/apikey 접속
        2. "Create API Key" 클릭
        3. API 키 복사 (AIza...로 시작)
        
        **Streamlit Cloud 배포 시:**
        Settings → Secrets → 아래 내용 붙여넣기
        ```
        GOOGLE_API_KEY = "AIza..."
        ```
        
        **로컬 개발 시:**
        .streamlit/secrets.toml 파일 생성
        ```
        GOOGLE_API_KEY = "AIza..."
        ```
        """)
        return None
    
    # 여러 모델명 시도 (Gemini 2.5 최신!)
    models_to_try = [
        "models/gemini-2.5-flash",  # ⭐ 최신 2.5 Flash (가장 빠름!)
        "models/gemini-2.5-pro-preview-05-06",  # 2.5 Pro
        "gemini-2.0-flash-exp",  # 2.0 실험 버전
        "models/gemini-1.5-flash-latest",  # 1.5 Flash (폴백)
        "models/gemini-1.5-pro-latest",  # 1.5 Pro (폴백)
    ]
    
    for model_name in models_to_try:
        try:
            print(f"시도 중: {model_name}")
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0,
                google_api_key=api_key
            )
            
            # 간단한 테스트
            test = llm.invoke("Hi")
            print(f"✅ 성공: {model_name}")
            st.success(f"✅ Google Gemini ({model_name}) 로드 완료!")
            return llm
            
        except Exception as e:
            print(f"❌ 실패: {model_name} - {e}")
            continue
    
    # 모든 모델 실패
    st.warning(f"""
    ⚠️ Gemini 로드 실패: 모든 모델을 시도했지만 실패했습니다.
    
    **시도한 모델:**
    {', '.join(models_to_try)}
    
    **임시 해결:** AI 인사이트 없이 기본 분석 모드로 사용 가능합니다.
    """)
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
                except:
                    pass
        
        # 숫자 변환
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    converted = pd.to_numeric(df[col], errors='coerce')
                    if not converted.isna().all():
                        df[col] = converted
                except:
                    pass
        
        df.dropna(how='all', inplace=True)
        st.success(f"✅ 로드 완료: {len(df):,}행 × {len(df.columns)}컬럼")
        return df
    except Exception as e:
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
        df_facility = pd.DataFrame({
            'wrk_date': dates,
            'md_shft': np.random.choice(['A', 'B', 'C'], 2631),
            'prod_wgt': np.random.normal(1500, 300, 2631),
            'eaf_wat_sum': np.random.normal(5000, 1000, 2631),
            'wat_unit': np.where(
                pd.to_datetime(dates).month == 7,
                np.random.normal(630, 30, 2631),
                np.random.normal(450, 50, 2631)
            )
        })
        st.success("✅ 샘플 데이터 생성!")
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
    
    # --- AI 질의응답 ---
    st.divider()
    st.subheader("💬 AI에게 질문하기")
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    sample_qs = {
        "Q1": "prod_wgt 평균은?",
        "Q2": "wat_unit 월별 추이",
        "Q3": "md_shft별 wat_unit 추이",
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
        placeholder="예: md_shft별로 wat_unit 월별 추이를 보여줘"
    )
    
    if st.button("🚀 분석", type="primary"):
        if user_question:
            # 질문 유형 분석
            is_time_series = any(kw in user_question for kw in ["월별", "추이", "trend", "변화", "그래프"])
            is_multi_series = any(kw in user_question for kw in ["계열", "조로", "구분", "별로", "분리", "그룹별"])
            
            # 날짜 컬럼 찾기
            date_col = None
            for col in df_facility.columns:
                if 'date' in col.lower() and pd.api.types.is_datetime64_any_dtype(df_facility[col]):
                    date_col = col
                    break
            
            # 분석 컬럼 찾기
            mentioned_col = None
            for col in numeric_cols:
                if col in user_question.lower():
                    mentioned_col = col
                    break
            
            # 그룹 컬럼 찾기
            group_col = None
            if is_multi_series:
                cat_cols = df_facility.select_dtypes(include=['object']).columns
                for col in cat_cols:
                    if col in user_question.lower():
                        group_col = col
                        break
            
            # 간단한 질문 직접 처리
            if "행" in user_question or "row" in user_question.lower():
                st.success(f"데이터 행 수: **{len(df_facility):,}개**")
            
            elif "컬럼" in user_question:
                st.success(f"컬럼: {', '.join(df_facility.columns.tolist())}")
            
            elif "평균" in user_question and mentioned_col:
                avg = df_facility[mentioned_col].mean()
                st.success(f"{mentioned_col} 평균: **{avg:,.2f}**")
            
            elif "합계" in user_question and mentioned_col:
                total = df_facility[mentioned_col].sum()
                st.success(f"{mentioned_col} 합계: **{total:,.2f}**")
            
            elif "최댓값" in user_question and mentioned_col:
                max_val = df_facility[mentioned_col].max()
                st.success(f"{mentioned_col} 최댓값: **{max_val:,.2f}**")
            
            elif "결측치" in user_question:
                null_cols = df_facility.isnull().sum()
                null_cols = null_cols[null_cols > 0]
                if len(null_cols) > 0:
                    st.write("**결측치가 있는 컬럼:**")
                    st.dataframe(null_cols)
                else:
                    st.success("결측치가 없습니다!")
            
            # 월별 추이 분석
            elif is_time_series and date_col and mentioned_col:
                if is_multi_series and group_col:
                    # === 다중 계열 분석 ===
                    st.markdown("### 📈 계열별 월별 추이 그래프")
                    
                    temp_df = df_facility.copy()
                    temp_df['month'] = temp_df[date_col].dt.month
                    multi = temp_df.groupby(['month', group_col])[mentioned_col].mean().reset_index()
                    multi.columns = ['월', group_col, mentioned_col]
                    
                    fig = px.line(multi, x='월', y=mentioned_col, color=group_col,
                                markers=True, title=f'{mentioned_col}의 {group_col}별 월별 추이')
                    fig.update_xaxes(title="월", dtick=1)
                    fig.update_yaxes(title=f"{mentioned_col} 평균")
                    fig.update_layout(legend_title=group_col)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    with st.expander("📊 계열별 데이터 테이블"):
                        pivot = multi.pivot(index='월', columns=group_col, values=mentioned_col)
                        st.dataframe(pivot)
                    
                    # 프로세스 표시
                    process = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "question_type": "계열별 월별 추이",
                        "date_column": date_col,
                        "value_column": mentioned_col,
                        "group_column": group_col,
                        "groups": df_facility[group_col].unique().tolist(),
                        "steps": [
                            f"✅ 질문 유형: 계열별 월별 추이",
                            f"✅ 날짜 컬럼: {date_col}",
                            f"✅ 분석 컬럼: {mentioned_col}",
                            f"✅ 그룹 컬럼: {group_col}",
                            f"✅ 월 추출: df['{date_col}'].dt.month",
                            f"✅ 그룹화: groupby(['month', '{group_col}'])['{mentioned_col}'].mean()",
                            "✅ 다중 선그래프 생성"
                        ],
                        "pandas_code": f"df.groupby([df['{date_col}'].dt.month, '{group_col}'])['{mentioned_col}'].mean()",
                        "plotly_code": f"px.line(data, x='월', y='{mentioned_col}', color='{group_col}', markers=True)"
                    }
                    
                    with st.expander("🔍 분석 프로세스"):
                        st.write("**📋 실행 단계:**")
                        for step in process["steps"]:
                            st.write(step)
                        
                        st.write("\n**💻 Pandas 코드:**")
                        st.code(process["pandas_code"], language="python")
                        
                        st.write("\n**💻 Plotly 코드:**")
                        st.code(process["plotly_code"], language="python")
                        
                        st.write("\n**🔧 전체 프로세스:**")
                        st.json(process)
                    
                    # 계열별 인사이트
                    st.markdown("### 🎯 계열별 핵심 인사이트")
                    
                    for group in df_facility[group_col].unique():
                        group_data = multi[multi[group_col] == group]
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
                    
                    # AI 인사이트 (LLM 있을 때만)
                    if llm:
                        with st.spinner("AI 인사이트 생성 중..."):
                            try:
                                prompt = f"""
다음은 {group_col}별 {mentioned_col}의 월별 평균 데이터입니다:
{multi.to_string()}

철강 설비 데이터 전문가로서, 각 그룹별로 주목할 만한 특징 2-3가지를 한국어로 간단히 설명하세요.
"""
                                insight = llm.invoke(prompt)
                                st.success(f"**🤖 AI 인사이트:**\n\n{insight.content}")
                            except Exception as e:
                                st.warning(f"AI 인사이트 생성 실패: {e}")
                
                else:
                    # === 단일 계열 분석 ===
                    st.markdown("### 📈 월별 추이")
                    
                    temp_df = df_facility.copy()
                    temp_df['month'] = temp_df[date_col].dt.month
                    monthly = temp_df.groupby('month')[mentioned_col].mean().reset_index()
                    monthly.columns = ['월', mentioned_col]
                    
                    fig = px.line(monthly, x='월', y=mentioned_col,
                                markers=True, title=f'{mentioned_col}의 월별 평균 추이')
                    fig.update_xaxes(title="월", dtick=1)
                    fig.update_yaxes(title=f"{mentioned_col} 평균")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    with st.expander("📊 월별 데이터 테이블"):
                        st.dataframe(monthly)
                    
                    # 프로세스 표시
                    process = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "question_type": "월별 추이",
                        "date_column": date_col,
                        "value_column": mentioned_col,
                        "steps": [
                            f"✅ 질문 유형: 월별 추이",
                            f"✅ 날짜 컬럼: {date_col}",
                            f"✅ 분석 컬럼: {mentioned_col}",
                            f"✅ 월 추출: df['{date_col}'].dt.month",
                            f"✅ 그룹화: groupby('month')['{mentioned_col}'].mean()",
                            "✅ 선그래프 생성"
                        ],
                        "pandas_code": f"df.groupby(df['{date_col}'].dt.month)['{mentioned_col}'].mean()",
                        "plotly_code": f"px.line(monthly, x='월', y='{mentioned_col}', markers=True)"
                    }
                    
                    with st.expander("🔍 분석 프로세스"):
                        st.write("**📋 실행 단계:**")
                        for step in process["steps"]:
                            st.write(step)
                        
                        st.write("\n**💻 Pandas 코드:**")
                        st.code(process["pandas_code"], language="python")
                        
                        st.write("\n**💻 Plotly 코드:**")
                        st.code(process["plotly_code"], language="python")
                        
                        st.write("\n**🔧 전체 프로세스:**")
                        st.json(process)
                    
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
                    
                    # AI 추가 인사이트
                    if llm:
                        with st.spinner("AI 분석 중..."):
                            try:
                                prompt = f"""
{mentioned_col}의 월별 평균:
{monthly.to_string()}

이 데이터에서 주목할 만한 패턴이나 이상치를 2-3가지 한국어로 설명하세요.
"""
                                insight = llm.invoke(prompt)
                                st.success(f"**🤖 AI 분석:**\n\n{insight.content}")
                            except Exception as e:
                                st.warning(f"AI 분석 실패: {e}")
            
            else:
                st.warning("질문을 이해하지 못했습니다. 더 구체적으로 질문해주세요.")
            
            if 'sample_question' in st.session_state:
                del st.session_state.sample_question
        else:
            st.warning("질문을 입력하세요")
    
    # --- 수동 그래프 ---
    st.divider()
    st.subheader("📈 수동 그래프 생성")
    
    col1, col2 = st.columns(2)
    with col1:
        chart_type = st.selectbox("차트", ["선그래프", "막대", "히스토그램", "박스플롯"])
    with col2:
        if numeric_cols:
            selected_col = st.selectbox("컬럼", numeric_cols)
    
    if st.button("📊 그래프 생성", type="secondary"):
        if selected_col:
            if chart_type == "선그래프":
                fig = px.line(df_facility.head(100), y=selected_col, title=selected_col)
            elif chart_type == "막대":
                fig = px.bar(df_facility.head(50), y=selected_col, title=selected_col)
            elif chart_type == "히스토그램":
                fig = px.histogram(df_facility, x=selected_col, title=f"{selected_col} 분포")
            else:
                fig = px.box(df_facility, y=selected_col, title=selected_col)
            
            st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption("🔧 철강 설비 AI 대시보드 v8.0 Final | Streamlit + Gemini 2.5")
