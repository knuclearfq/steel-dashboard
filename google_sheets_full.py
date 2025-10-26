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
    """1단계: 히스토리 기본 정보만 (ID, 타임스탬프, 질문, 시간단위, 그래프타입)"""
    try:
        response = requests.get(
            web_app_url,
            params={
                "api_key": api_key,
                "mode": "summary"
            },
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
                error_msg = result.get('error', 'Unknown error')
                debug_info = result.get('debug', {})
                
                if debug_info:
                    st.error(f"❌ API 에러: {error_msg}")
                    st.error(f"🔑 받은 키: {debug_info.get('received', 'N/A')}")
                    st.error(f"🔑 예상 키: {debug_info.get('expected', 'N/A')}")
                    st.info("💡 Streamlit secrets와 Apps Script의 API 키가 일치하는지 확인하세요!")
                else:
                    st.error(f"❌ API 에러: {error_msg}")
                
                return pd.DataFrame()
        else:
            st.error(f"❌ HTTP 에러: {response.status_code}")
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"❌ 연결 실패: {e}")
        return pd.DataFrame()

def load_graph_by_id(web_app_url, api_key, history_id):
    """2단계: 특정 ID의 그래프만 조회"""
    try:
        response = requests.get(
            web_app_url,
            params={
                "api_key": api_key,
                "mode": "graph",
                "id": history_id
            },
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get("success"):
                return result.get("data")
            else:
                return None
        else:
            return None
            
    except Exception as e:
        st.error(f"❌ 그래프 로딩 실패: {e}")
        return None

@st.cache_data(ttl=60)
def load_full_history(web_app_url, api_key):
    """3단계: 전체 히스토리 불러오기 (상세정보 - 모든 데이터)"""
    try:
        response = requests.get(
            web_app_url,
            params={
                "api_key": api_key,
                "mode": "full"  # 전체 데이터
            },
            timeout=30
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
        
        # POST 데이터 준비 (API 키는 제외)
        payload = {
            "id": history_entry.get("id", "")[:8],
            "timestamp": history_entry.get("timestamp", ""),
            "question": history_entry.get("question", ""),
            "result_type": history_entry.get("result_type", ""),
            "time_unit": history_entry.get("time_unit", ""),
            "chart_type": history_entry.get("chart_type", ""),
            "insights": (history_entry.get("insights", "") or "")[:20000],
            "data_json": data_json,
            "data_code": history_entry.get("data_code", ""),
            "graph_code": history_entry.get("code", ""),
            "graph_json": graph_config_json
        }
        
        # API 키는 URL parameter로 전달 (Apps Script의 e.parameter에서 읽음)
        response = requests.post(
            web_app_url,
            params={"api_key": api_key},  # ✅ URL parameter로!
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get("success"):
                return True, "저장 성공"
            else:
                error_msg = result.get('error', 'Unknown error')
                debug_info = result.get('debug', {})
                
                if debug_info:
                    return False, f"저장 실패: {error_msg}\n🔑 받은: {debug_info.get('received')}\n🔑 예상: {debug_info.get('expected')}"
                else:
                    return False, f"저장 실패: {error_msg}"
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
    """완전 재현 시스템 UI - 그래프는 항상 표시, 상세 정보는 토글"""
    
    st.divider()
    st.markdown("## 📊 분석 히스토리")
    
    # 설정 확인
    web_app_url, api_key = get_apps_script_config()
    
    if not web_app_url or not api_key:
        st.warning("⚠️ Apps Script 설정이 필요합니다")
        return
    
    # === 통계 요약 (항상 표시) ===
    st.markdown("### 📈 저장 통계")
    
    try:
        stats_df = load_history_summary(web_app_url, api_key)
        
        if not stats_df.empty:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("총 분석 수", f"{len(stats_df):,}")
            
            with col2:
                if "그래프타입" in stats_df.columns:
                    most_common = stats_df["그래프타입"].mode()[0] if not stats_df["그래프타입"].mode().empty else "N/A"
                    st.metric("가장 많이 사용한 그래프", most_common)
            
            with col3:
                if "시간단위" in stats_df.columns:
                    most_common_unit = stats_df["시간단위"].mode()[0] if not stats_df["시간단위"].mode().empty else "N/A"
                    st.metric("가장 많이 사용한 시간단위", most_common_unit)
            
            # 그래프 타입 분포 (항상 표시)
            if "그래프타입" in stats_df.columns:
                st.markdown("#### 📊 그래프 타입 분포")
                chart_counts = stats_df["그래프타입"].value_counts()
                
                # 막대 차트로 표시
                fig = go.Figure(data=[
                    go.Bar(
                        x=chart_counts.index,
                        y=chart_counts.values,
                        text=chart_counts.values,
                        textposition='auto',
                    )
                ])
                fig.update_layout(
                    height=300,
                    xaxis_title="그래프 타입",
                    yaxis_title="사용 횟수",
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True, key="stats_chart")
        else:
            st.info("📭 아직 저장된 분석이 없습니다")
    
    except Exception as e:
        st.error(f"❌ 통계 로딩 실패: {e}")
    
    st.divider()
    
    # === 히스토리 목록 (항상 표시) ===
    st.markdown("## 📋 분석 히스토리")
    
    try:
        # 1단계: 기본 정보 로딩
        stats_df = load_history_summary(web_app_url, api_key)
        
        if stats_df.empty:
            st.info("📭 아직 저장된 분석이 없습니다")
            return
        
        # 2단계: 모든 그래프 한 번에 로딩
        with st.spinner("📊 그래프 로딩 중..."):
            all_ids = stats_df['ID'].tolist() if 'ID' in stats_df.columns else []
            graph_map = {}
            
            for history_id in all_ids:
                if history_id:
                    graph_data = load_graph_by_id(web_app_url, api_key, history_id)
                    if graph_data:
                        graph_map[history_id] = graph_data.get('그래프_설정_JSON', '')
        
        # 각 히스토리 항목 표시 (토글 없이)
        for idx, row in stats_df.iterrows():
            history_id = row.get('ID', '')
            timestamp = row.get('타임스탬프', '')
            question = row.get('질문', '')
            time_unit = row.get('시간단위', '')
            chart_type = row.get('그래프타입', '')
            
            if not history_id:
                continue
            
            # 구분선
            if idx > 0:
                st.divider()
            
            # 히스토리 제목
            st.markdown(f"### 📊 {question}")
            
            # 기본 정보
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption(f"🕐 {timestamp}")
            with col2:
                st.caption(f"📈 {chart_type}")
            with col3:
                st.caption(f"⏱️ {time_unit}")
            
            # 그래프 (항상 표시) - 고유 key 사용
            if history_id in graph_map and graph_map[history_id]:
                try:
                    graph_json = graph_map[history_id]
                    fig = go.Figure(json.loads(graph_json))
                    st.plotly_chart(fig, use_container_width=True, key=f"graph_{history_id}")
                except Exception as e:
                    st.error(f"그래프 로딩 실패: {e}")
            else:
                st.info("그래프 데이터 없음")
            
            # 개별 상세정보 체크박스
            show_detail = st.checkbox(
                "🔍 상세정보 보기 (데이터, 코드, 인사이트)",
                key=f"detail_{history_id}",
                value=False
            )
            
            # 3단계: 체크박스 클릭 시 해당 항목만 Full 로딩
            if show_detail:
                with st.spinner(f"상세정보 로딩 중..."):
                    full_data = load_full_history_by_id(web_app_url, api_key, history_id)
                    
                    if full_data:
                        tabs = st.tabs(["📋 데이터", "💻 코드", "💡 인사이트"])
                        
                        # 데이터 탭
                        with tabs[0]:
                            data_json = full_data.get('데이터_JSON', '')
                            if data_json:
                                try:
                                    data = json.loads(data_json)
                                    df = pd.DataFrame(data)
                                    st.dataframe(df, use_container_width=True, key=f"data_{history_id}")
                                    st.caption(f"총 {len(df):,}행")
                                except Exception as e:
                                    st.error(f"데이터 복원 실패: {e}")
                            else:
                                st.info("데이터 없음")
                        
                        # 코드 탭
                        with tabs[1]:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**🔧 데이터 처리 코드**")
                                data_code = full_data.get('데이터_처리_코드', '')
                                if data_code:
                                    st.code(data_code, language="python")
                                else:
                                    st.info("코드 없음")
                            
                            with col2:
                                st.markdown("**📊 그래프 생성 코드**")
                                graph_code = full_data.get('그래프_생성_코드', '')
                                if graph_code:
                                    st.code(graph_code, language="python")
                                else:
                                    st.info("코드 없음")
                        
                        # 인사이트 탭
                        with tabs[2]:
                            insights = full_data.get('인사이트요약', '')
                            if insights:
                                st.info(insights)
                            else:
                                st.info("인사이트 없음")
                    else:
                        st.error("상세정보를 불러올 수 없습니다")
    
    except Exception as e:
        st.error(f"❌ 히스토리 로딩 실패: {e}")

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
