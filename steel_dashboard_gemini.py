import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
import warnings
from datetime import datetime

# 경고 메시지 무시
warnings.filterwarnings('ignore')

# --- 페이지 설정 ---
st.set_page_config(page_title="철강 설비 AI 대시보드", layout="wide")
st.title("🤖 철강 설비 AI 에이전트 대시보드")

# --- Gemini LLM 로드 (클라우드 배포용) ---
@st.cache_resource
def get_llm():
    """Google Gemini LLM 객체를 로드하는 함수"""
    print("Gemini LLM 로드 시도...")
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        # Streamlit Secrets 또는 환경변수에서 API 키 가져오기
        api_key = st.secrets.get("GOOGLE_API_KEY") if "GOOGLE_API_KEY" in st.secrets else os.getenv("GOOGLE_API_KEY")
        
        if not api_key:
            st.error("""
            ❌ Google Gemini API 키가 설정되지 않았습니다.
            
            **API 키 발급 (무료):**
            1. https://makersuite.google.com/app/apikey 접속
            2. "Create API Key" 클릭
            3. API 키 복사 (AIza...로 시작)
            
            **Streamlit Cloud 배포 시:**
            - App 설정 → Secrets → 아래 내용 붙여넣기
            ```
            GOOGLE_API_KEY = "AIza..."
            ```
            
            **로컬 테스트 시:**
            - `.streamlit/secrets.toml` 파일 생성
            - 위 내용 붙여넣기
            """)
            return None
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0,
            google_api_key=api_key,
            convert_system_message_to_human=True
        )
        
        # 간단한 연결 테스트
        test = llm.invoke("Hi")
        print(f"✅ Gemini 연결 성공")
        return llm
        
    except ImportError:
        st.error("""
        ❌ langchain-google-genai 패키지가 설치되지 않았습니다.
        
        **해결:**
        `requirements.txt`에 다음 추가:
        ```
        langchain-google-genai>=1.0.0
        ```
        """)
        return None
    except Exception as e:
        print(f"Gemini LLM 로드 실패: {e}")
        st.error(f"❌ Gemini 로드 실패: {e}")
        return None

llm = get_llm()
if llm:
    st.success("✅ Google Gemini Pro 로드 완료 (무료 할당량 사용)")
else:
    st.warning("⚠️ AI 분석 없이 기본 모드로 실행")

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
            if 'date' in col or 'dt' in col:
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
            'wat_unit': np.where(
                pd.to_datetime(dates).month == 7,
                np.random.normal(630, 30, 2631),
                np.random.normal(450, 50, 2631)
            )
        })
        st.success("✅ 샘플 데이터 생성!")

if df_facility is not None:
    with st.expander("🔍 데이터 미리보기"):
        st.dataframe(df_facility.head(10))
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**행 수:** {len(df_facility):,}")
        st.write(f"**컬럼 수:** {len(df_facility.columns)}")
    with col2:
        numeric_cols = df_facility.select_dtypes(include=['int64', 'float64']).columns.tolist()
        st.write(f"**수치형:** {len(numeric_cols)}개")
        st.write(f"**범주형:** {len(df_facility.select_dtypes(include=['object']).columns)}개")
    
    # --- AI 질의응답 ---
    st.divider()
    st.subheader("💬 AI에게 질문하기")
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    sample_qs = {
        "Q1": "prod_wgt 평균은?",
        "Q2": "wat_unit 월별 추이",
        "Q3": "md_shft별 wat_unit 추이",
    }
    
    cols = st.columns(3)
    for idx, (key, q) in enumerate(sample_qs.items()):
        if cols[idx].button(key, help=q):
            st.session_state.sample_question = q
    
    user_question = st.text_input(
        "질문:",
        value=st.session_state.get('sample_question', ''),
        placeholder="예: wat_unit의 월별 추이를 보여줘"
    )
    
    if st.button("🚀 AI 분석", type="primary"):
        if user_question:
            # 질문 유형 분석
            is_time_series = any(kw in user_question for kw in ["월별", "추이", "trend", "변화", "그래프"])
            is_multi_series = any(kw in user_question for kw in ["계열", "조로", "구분", "별로", "분리"])
            
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
                answer = f"데이터 행 수: **{len(df_facility):,}개**"
                st.success(answer)
            
            elif "컬럼" in user_question:
                answer = f"컬럼: {', '.join(df_facility.columns.tolist())}"
                st.success(answer)
            
            elif "평균" in user_question and mentioned_col:
                avg = df_facility[mentioned_col].mean()
                st.success(f"{mentioned_col} 평균: **{avg:,.2f}**")
            
            # 월별 추이 분석
            elif is_time_series and date_col and mentioned_col:
                if is_multi_series and group_col:
                    # 다중 계열
                    st.markdown("### 📈 계열별 월별 추이")
                    
                    temp_df = df_facility.copy()
                    temp_df['month'] = temp_df[date_col].dt.month
                    multi = temp_df.groupby(['month', group_col])[mentioned_col].mean().reset_index()
                    
                    fig = px.line(multi, x='month', y=mentioned_col, color=group_col,
                                markers=True, title=f'{mentioned_col}의 {group_col}별 월별 추이')
                    fig.update_xaxes(title="월", dtick=1)
                    fig.update_yaxes(title=f"{mentioned_col} 평균")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    with st.expander("📊 데이터 테이블"):
                        pivot = multi.pivot(index='month', columns=group_col, values=mentioned_col)
                        st.dataframe(pivot)
                    
                    # 계열별 인사이트
                    if llm:
                        with st.spinner("AI 인사이트 생성 중..."):
                            try:
                                prompt = f"""
다음은 {group_col}별 {mentioned_col}의 월별 평균 데이터입니다:
{multi.to_string()}

철강 설비 데이터 전문가로서, 각 그룹별로 주목할 만한 특징 2가지를 한국어로 간단히 설명하세요.
"""
                                insight = llm.invoke(prompt)
                                st.info(f"**🎯 AI 인사이트:**\n\n{insight.content}")
                            except:
                                pass
                
                else:
                    # 단일 계열
                    st.markdown("### 📈 월별 추이")
                    
                    temp_df = df_facility.copy()
                    temp_df['month'] = temp_df[date_col].dt.month
                    monthly = temp_df.groupby('month')[mentioned_col].mean().reset_index()
                    
                    fig = px.line(monthly, x='month', y=mentioned_col,
                                markers=True, title=f'{mentioned_col}의 월별 평균 추이')
                    fig.update_xaxes(title="월", dtick=1)
                    fig.update_yaxes(title=f"{mentioned_col} 평균")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    max_month = monthly.loc[monthly[mentioned_col].idxmax(), 'month']
                    max_val = monthly[mentioned_col].max()
                    min_month = monthly.loc[monthly[mentioned_col].idxmin(), 'month']
                    min_val = monthly[mentioned_col].min()
                    
                    st.info(f"""
**🎯 핵심 인사이트:**
- 최고점: {int(max_month)}월 ({max_val:,.2f})
- 최저점: {int(min_month)}월 ({min_val:,.2f})
- 변동폭: {max_val - min_val:,.2f}
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
                                st.success(f"**AI 분석:**\n\n{insight.content}")
                            except:
                                pass
            
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
st.caption("🔧 철강 설비 AI 대시보드 v7.0 | Streamlit + Gemini")
