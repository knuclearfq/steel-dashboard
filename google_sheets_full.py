"""
🔗 Google Sheets 완전 재현 시스템

데이터, 코드, 그래프 모두 저장하여 완벽하게 재현 가능
"""

import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import uuid
import json
import plotly.graph_objects as go
import plotly.express as px

# ============================================
# ⚙️ 설정
# ============================================

def get_apps_script_config():
    """Apps Script 설정 가져오기"""
    if "google_apps_script" not in st.secrets:
        return None, None
    
    config = st.secrets["google_apps_script"]
    web_app_url = config.get("web_app_url")
    api_key = config.get("api_key")
    
    return web_app_url, api_key

# ============================================
# 📖 히스토리 읽기 (요약)
# ============================================

@st.cache_data(ttl=60)
def load_history_summary(web_app_url, api_key):
    """히스토리 요약 목록 불러오기"""
    try:
        response = requests.get(
            web_app_url,
            params={"api_key": api_key},
            timeout=15
        )
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get("success"):
                data = result.get("data", [])
                if data:
                    return pd.DataFrame(data)
                else:
                    return pd.DataFrame()
            else:
                st.error(f"❌ API 에러: {result.get('error')}")
                return pd.DataFrame()
        else:
            st.error(f"❌ HTTP 에러: {response.status_code}")
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"❌ 연결 실패: {e}")
        return pd.DataFrame()

# ============================================
# 🔍 특정 ID의 완전한 데이터 조회
# ============================================

def load_full_history_by_id(web_app_url, api_key, history_id):
    """특정 ID의 완전한 히스토리 불러오기 (데이터, 코드 포함)"""
    try:
        response = requests.get(
            web_app_url,
            params={
                "api_key": api_key,
                "id": history_id
            },
            timeout=20
        )
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get("success"):
                return result.get("data")
            else:
                st.error(f"❌ API 에러: {result.get('error')}")
                return None
        else:
            st.error(f"❌ HTTP 에러: {response.status_code}")
            return None
            
    except Exception as e:
        st.error(f"❌ 연결 실패: {e}")
        return None

# ============================================
# ✍️ 완전한 히스토리 저장
# ============================================

def save_full_history(web_app_url, api_key, history_entry):
    """
    완전한 히스토리 저장 (데이터, 코드 모두 포함)
    
    history_entry에 포함될 것:
    - 기본 정보 (id, timestamp, question, result_type, time_unit, chart_type, insights)
    - data: DataFrame 또는 dict
    - data_processing_code: 데이터 처리 Python 코드
    - graph_code: 그래프 생성 Python 코드
    - figure: Plotly figure 객체
    """
    try:
        # 데이터를 JSON으로 변환
        data_json = None
        if history_entry.get("data") is not None:
            if isinstance(history_entry["data"], pd.DataFrame):
                data_json = history_entry["data"].to_json(orient="records")
            elif isinstance(history_entry["data"], dict):
                data_json = json.dumps(history_entry["data"])
        
        # Plotly figure를 JSON으로 변환
        graph_config_json = None
        if history_entry.get("figure") is not None:
            try:
                graph_config_json = history_entry["figure"].to_json()
            except:
                pass
        
        # POST 데이터 준비
        payload = {
            "action": "add",
            "api_key": api_key,
            "id": history_entry.get("id", "")[:8],
            "timestamp": history_entry.get("timestamp", ""),
            "question": history_entry.get("question", ""),
            "result_type": history_entry.get("result_type", ""),
            "time_unit": history_entry.get("time_unit", ""),
            "chart_type": history_entry.get("chart_type", ""),
            "insights": (history_entry.get("insights", "") or "")[:500],
            "data_json": data_json,
            "data_processing_code": history_entry.get("data_code", ""),
            "graph_code": history_entry.get("code", ""),
            "graph_config_json": graph_config_json
        }
        
        response = requests.post(
            web_app_url,
            json=payload,
            timeout=30  # 큰 데이터를 위해 timeout 증가
        )
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get("success"):
                return True, "저장 성공"
            else:
                return False, f"저장 실패: {result.get('error')}"
        else:
            return False, f"HTTP 에러: {response.status_code}"
            
    except Exception as e:
        return False, f"연결 실패: {e}"

# ============================================
# 🔄 히스토리 재현
# ============================================

def reproduce_history(full_history_data):
    """
    저장된 히스토리를 완전히 재현
    
    Args:
        full_history_data: load_full_history_by_id()의 결과
    
    Returns:
        재현된 figure, data, insights
    """
    try:
        st.markdown("### 🔄 히스토리 재현 중...")
        
        # 1. 데이터 복원
        data_df = None
        if full_history_data.get("데이터_JSON"):
            try:
                data_json = full_history_data["데이터_JSON"]
                if data_json and not data_json.endswith("[TRUNCATED]"):
                    data_df = pd.read_json(data_json)
                    st.success(f"✅ 데이터 복원: {len(data_df)}행 × {len(data_df.columns)}열")
                else:
                    st.warning("⚠️ 데이터가 너무 커서 일부만 저장되었습니다")
            except Exception as e:
                st.error(f"❌ 데이터 복원 실패: {e}")
        
        # 2. 그래프 복원
        figure = None
        if full_history_data.get("그래프_설정_JSON"):
            try:
                graph_json = full_history_data["그래프_설정_JSON"]
                if graph_json and not graph_json.endswith("[TRUNCATED]"):
                    figure = go.Figure(json.loads(graph_json))
                    st.success("✅ 그래프 복원 완료")
                else:
                    st.warning("⚠️ 그래프 설정이 너무 커서 일부만 저장되었습니다")
            except Exception as e:
                st.error(f"❌ 그래프 복원 실패: {e}")
        
        # 3. 코드 표시
        tabs = st.tabs(["📊 그래프", "📋 데이터", "💻 코드"])
        
        with tabs[0]:
            if figure:
                st.plotly_chart(figure, use_container_width=True)
            else:
                st.warning("그래프를 복원할 수 없습니다")
        
        with tabs[1]:
            if data_df is not None:
                st.dataframe(data_df, use_container_width=True)
            else:
                st.warning("데이터를 복원할 수 없습니다")
        
        with tabs[2]:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 그래프 생성 코드")
                graph_code = full_history_data.get("그래프_생성_코드", "")
                if graph_code:
                    st.code(graph_code, language="python")
                else:
                    st.info("코드 없음")
            
            with col2:
                st.markdown("#### 🔧 데이터 처리 코드")
                data_code = full_history_data.get("데이터_처리_코드", "")
                if data_code:
                    st.code(data_code, language="python")
                else:
                    st.info("코드 없음")
        
        # 4. 인사이트
        insights = full_history_data.get("인사이트요약", "")
        if insights:
            st.markdown("### 💡 인사이트")
            st.info(insights)
        
        return figure, data_df, insights
        
    except Exception as e:
        st.error(f"❌ 재현 실패: {e}")
        return None, None, None

# ============================================
# 🖥️ UI 구현
# ============================================

def render_full_history_ui():
    """완전 재현 시스템 UI"""
    
    st.divider()
    
    with st.expander("📊 Google Sheets 완전 히스토리 관리", expanded=False):
        st.markdown("""
        **🔄 완전 재현 시스템**
        
        저장되는 것:
        - ✅ 분석 결과 (기본 정보)
        - ✅ 데이터 테이블 (전체 DataFrame)
        - ✅ 데이터 처리 코드 (Python)
        - ✅ 그래프 생성 코드 (Plotly)
        - ✅ 그래프 설정 (JSON)
        - ✅ 인사이트
        
        → **언제든지 완벽하게 재현 가능!**
        """)
        
        # 설정 확인
        web_app_url, api_key = get_apps_script_config()
        
        if not web_app_url or not api_key:
            st.warning("⚠️ Apps Script 설정이 필요합니다")
            return
        
        st.success("✅ Apps Script 연결됨")
        
        tab1, tab2, tab3 = st.tabs(["📖 히스토리 목록", "🔄 히스토리 재현", "📊 통계"])
        
        # === 탭 1: 히스토리 목록 ===
        with tab1:
            col1, col2 = st.columns([1, 4])
            
            with col1:
                if st.button("🔄 새로고침", type="primary", key="refresh_full"):
                    st.cache_data.clear()
                    st.rerun()
            
            with col2:
                st.caption("💡 저장된 모든 히스토리")
            
            # 데이터 로드
            with st.spinner("📡 불러오는 중..."):
                history_df = load_history_summary(web_app_url, api_key)
            
            if not history_df.empty:
                st.success(f"✅ 총 {len(history_df)}개의 완전 히스토리")
                
                # 검색
                search_query = st.text_input(
                    "🔍 검색", 
                    placeholder="질문, 그래프 타입 등",
                    key="search_full"
                )
                
                if search_query:
                    mask = history_df.astype(str).apply(
                        lambda row: row.str.contains(search_query, case=False, na=False).any(), 
                        axis=1
                    )
                    filtered_df = history_df[mask]
                    st.write(f"**검색 결과: {len(filtered_df)}개**")
                else:
                    filtered_df = history_df
                
                # 표시
                st.dataframe(
                    filtered_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "ID": st.column_config.TextColumn("ID", width="small"),
                        "타임스탬프": st.column_config.TextColumn("시간", width="medium"),
                        "질문": st.column_config.TextColumn("질문", width="large"),
                    }
                )
                
            else:
                st.info("📭 저장된 히스토리가 없습니다")
        
        # === 탭 2: 히스토리 재현 ===
        with tab2:
            st.markdown("### 🔄 저장된 히스토리 완전 재현")
            
            # 세션 상태 초기화
            if 'selected_reproduce_id' not in st.session_state:
                st.session_state.selected_reproduce_id = ""
            
            # ID 입력 (세션 상태에서 기본값 가져오기)
            history_id = st.text_input(
                "히스토리 ID 입력",
                value=st.session_state.selected_reproduce_id,
                placeholder="예: 550e8400",
                help="재현할 히스토리의 ID를 입력하세요",
                key="reproduce_id_input"
            )
            
            if st.button("🔄 재현하기", type="primary", key="reproduce_btn"):
                if history_id:
                    with st.spinner("🔄 재현 중..."):
                        full_data = load_full_history_by_id(web_app_url, api_key, history_id)
                    
                    if full_data:
                        st.markdown(f"### 📋 히스토리: {full_data.get('질문', 'N/A')}")
                        st.caption(f"🕐 {full_data.get('타임스탬프', 'N/A')}")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("결과 타입", full_data.get('결과타입', 'N/A'))
                        with col2:
                            st.metric("그래프", full_data.get('그래프타입', 'N/A'))
                        with col3:
                            st.metric("시간 단위", full_data.get('시간단위', 'N/A'))
                        
                        st.divider()
                        
                        # 완전 재현
                        reproduce_history(full_data)
                    else:
                        st.error("❌ 히스토리를 찾을 수 없습니다")
                else:
                    st.warning("⚠️ ID를 입력하세요")
            
            # 최근 히스토리 목록
            st.divider()
            st.markdown("#### 📋 최근 히스토리")
            
            recent_df = load_history_summary(web_app_url, api_key)
            if not recent_df.empty:
                recent_5 = recent_df.head(5)
                
                for idx, row in recent_5.iterrows():
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        st.write(f"**{row['타임스탬프']}** - {row['질문']}")
                        st.caption(f"ID: `{row['ID']}` | {row['그래프타입']} | {row['시간단위']}")
                    
                    with col2:
                        if st.button("🔄 재현", key=f"quick_reproduce_{row['ID']}"):
                            st.session_state.selected_reproduce_id = row['ID']
                            st.rerun()
        
        # === 탭 3: 통계 ===
        with tab3:
            st.markdown("### 📊 저장 통계")
            
            stats_df = load_history_summary(web_app_url, api_key)
            
            if not stats_df.empty:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("총 히스토리", len(stats_df))
                
                with col2:
                    if "그래프타입" in stats_df.columns:
                        most_common = stats_df["그래프타입"].mode()[0] if not stats_df["그래프타입"].mode().empty else "N/A"
                        st.metric("인기 그래프", most_common)
                
                with col3:
                    if "시간단위" in stats_df.columns:
                        most_common_unit = stats_df["시간단위"].mode()[0] if not stats_df["시간단위"].mode().empty else "N/A"
                        st.metric("인기 시간단위", most_common_unit)
                
                # 분포 차트
                if "그래프타입" in stats_df.columns:
                    st.markdown("#### 📊 그래프 타입 분포")
                    chart_counts = stats_df["그래프타입"].value_counts()
                    st.bar_chart(chart_counts)
            else:
                st.info("📭 통계 데이터가 없습니다")

# ============================================
# 🔗 add_to_history 함수 (완전 버전)
# ============================================

def add_to_full_history(
    question, result_type, figure=None, data=None, 
    insights=None, code=None, data_code=None, 
    chart_type="line", time_unit="월별"
):
    """
    완전한 히스토리 저장 (데이터, 코드 모두 포함)
    
    이 함수를 기존 add_to_history 대신 사용하거나
    add_to_history 내부에서 호출
    """
    # 로컬 히스토리 저장
    history_entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "result_type": result_type,
        "figure": figure,
        "data": data,
        "insights": insights,
        "code": code,
        "data_code": data_code,
        "chart_type": chart_type,
        "time_unit": time_unit
    }
    
    st.session_state.analysis_history.append(history_entry)
    
    # Google Sheets에 완전 저장
    web_app_url, api_key = get_apps_script_config()
    
    if web_app_url and api_key:
        try:
            with st.spinner("💾 Google Sheets에 완전 히스토리 저장 중..."):
                success, message = save_full_history(
                    web_app_url, 
                    api_key, 
                    history_entry
                )
            
            if success:
                st.success(f"✅ {message} (ID: {history_entry['id'][:8]})")
                st.info("💡 나중에 이 ID로 완전히 재현할 수 있습니다!")
            else:
                st.warning(f"⚠️ {message}")
        except Exception as e:
            st.warning(f"⚠️ 저장 중 에러: {e}")

# ============================================
# 💡 사용 예시
# ============================================

"""
메인 대시보드 통합:

1. import
   from google_sheets_full import render_full_history_ui, add_to_full_history

2. UI 추가
   render_full_history_ui()

3. 히스토리 저장 (기존 add_to_history 대체)
   add_to_full_history(
       question=user_question,
       result_type="계열별_월별_추이",
       figure=fig,                  # Plotly figure
       data=multi_df,               # DataFrame
       insights=insights_text,
       code=graph_code,             # 그래프 생성 코드
       data_code=data_processing_code,  # 데이터 처리 코드
       chart_type="막대그래프",
       time_unit="월별"
   )

4. 재현
   히스토리 ID로 언제든지 완벽하게 재현 가능!
"""
