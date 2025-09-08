from typing import TypedDict, List, Optional, Dict, Literal

# 사용자 요청의 의도를 나타내는 타입 정의
Intent = Literal["template_generation", "legal_inquiry", "chit_chat", "anomalous_request"]

class GraphState(TypedDict):
    """
    그래프의 모든 노드에서 공유되는 상태 객체입니다.
    대화의 모든 정보를 포함합니다.
    """
    # --- 입력 및 분류 ---
    original_request: str               # 사용자의 최초 입력 메시지
    intent: Optional[Intent]            # 분류된 사용자의 의도
    
    # --- 파이프라인별 데이터 ---
    retrieved_docs: Optional[List[str]] # 법률 안내(RAG)를 통해 검색된 문서 내용
    
    # --- 템플릿 생성 파이프라인용 상태 ---
    # 기존 process_chat_message 함수가 사용하던 state를 내장
    template_pipeline_state: Dict       
    
    # --- 최종 결과 ---
    final_response: Optional[Dict]      # 사용자에게 전달될 최종 응답 데이터
    
    # --- LangGraph 내부 관리용 --- 
    # 이 필드는 LangGraph 내부에서만 사용되며, 최종 응답에는 포함되지 않습니다.
    error: Optional[str]                # 오류 발생 시 메시지 저장
    # 프론트엔드에 다음 액션을 지시하기 위한 필드 (LangGraph state에 직접 저장)
    next_action: Optional[str]          


