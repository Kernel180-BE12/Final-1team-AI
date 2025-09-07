from langgraph.graph import StateGraph, END
from state import GraphState
from nodes import (
    classify_intent_node,
    legal_inquiry_node,
    chit_chat_node,
    anomalous_request_node,
    template_generation_node,
    template_confirmation_node,
    cancel_node
)

def route_by_intent(state: GraphState) -> str:
    """의도에 따라 다음 노드를 결정하는 라우터 함수."""
    if state.get("error"):
        print("오류 발생, 그래프 종료")
        return END
    
    intent = state.get("intent")
    print(f"라우팅: 의도 '{intent}'에 따라 경로를 결정합니다.")
    
    # "예/아니오" 응답에 대한 라우팅
    if intent == "user_confirmed":
        return "template_generation_node"
    elif intent == "user_denied":
        return "cancel_node"
    elif intent == "confirmation_invalid":
        # "예", "아니오"가 아닌 다른 응답 시 다시 확인 요청
        return "template_confirmation_node"

    # 최초 요청의 의도에 대한 라우팅
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
workflow.add_node("classify_intent_node", classify_intent_node)
workflow.add_node("legal_inquiry_node", legal_inquiry_node)
workflow.add_node("chit_chat_node", chit_chat_node)
workflow.add_node("anomalous_request_node", anomalous_request_node)
workflow.add_node("template_generation_node", template_generation_node)
workflow.add_node("template_confirmation_node", template_confirmation_node)
workflow.add_node("cancel_node", cancel_node)


# 엣지(연결) 설정
workflow.set_entry_point("classify_intent_node")

# 의도 분류 및 확인 응답 처리가 끝난 후, 결과에 따라 분기
workflow.add_conditional_edges(
    "classify_intent_node",
    route_by_intent,
    {
        "template_confirmation_node": "template_confirmation_node",
        "template_generation_node": "template_generation_node",
        "cancel_node": "cancel_node",
        "legal_inquiry_node": "legal_inquiry_node",
        "chit_chat_node": "chit_chat_node",
        "anomalous_request_node": "anomalous_request_node",
        END: END
    }
)

# 확인/취소/생성 등 단일 행동 노드는 실행 후 종료됨
# 확인 노드는 사용자 입력을 기다리기 위해 일단 종료
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