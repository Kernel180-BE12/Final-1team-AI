from langgraph.graph import StateGraph, END
from state import GraphState
from nodes import (
    route_request_node,
    classify_intent_node,
    legal_inquiry_node,
    chit_chat_node,
    anomalous_request_node,
    template_generation_node,
    template_confirmation_node,
    cancel_node
)

def main_router(state: GraphState) -> str:
    """대화 상태를 기반으로 요청을 라우팅하는 메인 컨트롤러."""
    next_action = state.get("next_action")
    pipeline_step = state.get("template_pipeline_state", {}).get("step")
    if next_action == "awaiting_confirmation":
        user_response = state.get("original_request", "").strip()
        if user_response == "예":
            return "template_generation_node"
        elif user_response == "아니오":
            return "cancel_node"
        else:
            return "template_confirmation_node"
    
    elif pipeline_step and pipeline_step != 'initial':
        return "template_generation_node"
    
    else:
        return "classify_intent_node"

def route_by_intent(state: GraphState) -> str:
    """의도 분류 결과에 따라 다음 노드를 결정하는 라우터 함수."""
    if state.get("error"):
        return END
    
    intent = state.get("intent")
    print(f"라우팅: 의도 '{intent}'에 따라 경로를 결정합니다.")
    
    if intent == "template_generation":
        return "template_confirmation_node"
    elif intent == "legal_inquiry":
        return "legal_inquiry_node"
    elif intent == "chit_chat":
        return "chit_chat_node"
    elif intent == "anomalous_request":
        return "anomalous_request_node"
    else:
        return "chit_chat_node"

# LangGraph 워크플로우 정의
workflow = StateGraph(GraphState)

# 노드 추가
workflow.add_node("route_request_node", route_request_node)
workflow.add_node("classify_intent_node", classify_intent_node)
workflow.add_node("legal_inquiry_node", legal_inquiry_node)
workflow.add_node("chit_chat_node", chit_chat_node)
workflow.add_node("anomalous_request_node", anomalous_request_node)
workflow.add_node("template_generation_node", template_generation_node)
workflow.add_node("template_confirmation_node", template_confirmation_node)
workflow.add_node("cancel_node", cancel_node)


# 엣지(연결) 설정
workflow.set_entry_point("route_request_node")

# 1. 메인 라우터에서 각 상황에 맞게 분기
workflow.add_conditional_edges(
    "route_request_node",
    main_router,
    {
        "classify_intent_node": "classify_intent_node",
        "template_generation_node": "template_generation_node",
        "cancel_node": "cancel_node",
        "template_confirmation_node": "template_confirmation_node",
    }
)

# 2. 의도 분류 노드는 이제 새로운 요청만 처리
workflow.add_conditional_edges(
    "classify_intent_node",
    route_by_intent,
    {
        "template_confirmation_node": "template_confirmation_node",
        "legal_inquiry_node": "legal_inquiry_node",
        "chit_chat_node": "chit_chat_node",
        "anomalous_request_node": "anomalous_request_node",
        END: END
    }
)

# 3. 각 기능 노드는 실행 후 종료
workflow.add_edge("template_confirmation_node", END)
workflow.add_edge("cancel_node", END)
workflow.add_edge("legal_inquiry_node", END)
workflow.add_edge("chit_chat_node", END)
workflow.add_edge("anomalous_request_node", END)
workflow.add_edge("template_generation_node", END)


# 그래프 컴파일
app_graph = workflow.compile()

# (선택) 그래프 시각화
try:
    app_graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
    print("그래프 구조가 'graph.png' 파일로 저장되었습니다.")
except Exception as e:
    print(f"그래프 시각화 실패: {e} (pygraphviz가 설치되지 않았을 수 있습니다)")