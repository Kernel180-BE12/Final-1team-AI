from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Literal

from state import GraphState, Intent
from chatbot_logic import (
    initialize_system, 
    process_chat_message, 
    retrievers
)

# --- 모델 및 파서 초기화 ---
# 의도 분류 및 간단한 답변 생성에는 비용 효율적인 모델 사용
llm_fast = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
# 법률 자문과 같이 정확성이 중요한 작업에는 고성능 모델 사용
llm_smart = ChatOpenAI(model="gpt-4o", temperature=0)

class IntentClassifier(BaseModel):
    """사용자의 질문 의도를 분류합니다."""
    intent: Intent = Field(
        description="""사용자의 질문을 다음 네 가지 중 하나로 분류합니다:
- template_generation: 특정 목적의 알림톡 메시지 생성을 요청하는 경우 (예: "폐점 안내 메시지 만들어줘")
- legal_inquiry: 법률, 규정, 가이드라인에 대해 질문하는 경우 (예: "광고성 정보 기준이 뭐야?")
- chit_chat: 일반적인 인사, 안부, 농담 등 일상 대화
- anomalous_request: 비윤리적이거나, 폭력적이거나, 시스템 목적과 무관한 이상한 요청"""
    )

# --- 노드 함수 정의 ---

def classify_intent_node(state: GraphState) -> GraphState:
    """사용자의 요청 의도를 분류하거나, 기존 파이프라인을 계속 진행할지 결정합니다."""
    
    # [수정된 부분] 템플릿 생성 파이프라인이 진행 중인지 먼저 확인
    pipeline_state = state.get("template_pipeline_state", {})
    if pipeline_state and pipeline_state.get("step") != 'initial':
        print("--- 의도 분류 건너뛰기: 템플릿 생성 파이프라인 진행 중 ---")
        # 의도를 'template_generation'으로 강제하여 바로 다음 단계로 라우팅
        state["intent"] = "template_generation"
        return state

    print("--- 노드 실행: 의도 분류 ---")
    prompt = ChatPromptTemplate.from_template(
        "당신은 사용자의 메시지를 분석하여 가장 적절한 의도를 분류하는 전문가입니다. 사용자의 메시지에 대한 의도를 JSON 형식으로만 답변해주세요.\n사용자 메시지: {request}"
    )
    # 중요: 의도 분류는 시스템의 첫 관문이므로 정확도가 높은 모델 사용
    classifier_chain = prompt | llm_smart.with_structured_output(IntentClassifier)
    
    try:
        result = classifier_chain.invoke({"request": state["original_request"]})
        state["intent"] = result.intent
        print(f"-> 분류된 의도: {result.intent}")
    except Exception as e:
        print(f"오류: 의도 분류 실패 - {e}")
        state["error"] = f"의도 분류 중 오류 발생: {e}"
        state["intent"] = "chit_chat" # 오류 시 일상 대화로 처리하여 시스템 다운 방지
    return state

def legal_inquiry_node(state: GraphState) -> GraphState:
    """법률/가이드라인 정보를 검색하고 답변을 생성합니다."""
    print("--- 노드 실행: 법률/가이드라인 안내 ---")
    query = state["original_request"]
    
    # 'compliance' 리트리버가 있는지 확인 및 초기화 여부 검증
    if not retrievers or 'compliance' not in retrievers or not retrievers['compliance']:
        state["final_response"] = {"message": "죄송합니다. 법률 정보 데이터베이스가 현재 준비되지 않아 답변할 수 없습니다. 관리자에게 문의해주세요."}
        state["error"] = "Compliance retriever not initialized or found."
        return state
        
    try:
        docs = retrievers['compliance'].invoke(query)
        context = "\n\n".join([f"문서 {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
        state["retrieved_docs"] = [doc.page_content for doc in docs] # 디버깅용

        prompt = ChatPromptTemplate.from_template(
            """당신은 정보통신망법 및 알림톡 가이드라인 전문가입니다.
            아래에 제공된 '관련 규정'만을 근거로 사용자의 질문에 대해 정확하고 명료하게 답변해주세요.
            답변 시, 어떤 규정을 근거로 하였는지 간략히 언급해주세요. 추측성 답변은 절대 금지입니다.

            [관련 규정]
            {context}

            [사용자 질문]
            {query}

            [전문가 답변]
            """
        )
        response_chain = prompt | llm_smart | StrOutputParser()
        answer = response_chain.invoke({"context": context, "query": query})
        
        state["final_response"] = {"message": answer}
    except Exception as e:
        print(f"오류: 법률/가이드라인 안내 중 오류 발생 - {e}")
        state["error"] = f"법률/가이드라인 안내 중 오류 발생: {e}"
        state["final_response"] = {"message": "죄송합니다. 법률 정보를 처리하는 중 오류가 발생했습니다. 다시 시도해주세요."}
    return state

def chit_chat_node(state: GraphState) -> GraphState:
    """일상 대화에 대한 답변을 생성합니다."""
    print("--- 노드 실행: 일상 대화 ---")
    prompt = ChatPromptTemplate.from_template(
        "당신은 사용자의 비즈니스 메시징 업무를 돕는 친절하고 전문적인 AI 어시스턴트 'Manus'입니다. 사용자의 일상적인 대화에 대해 자연스럽고 긍정적으로 답변해주세요.\n사용자 메시지: {request}"
    )
    try:
        chain = prompt | llm_fast | StrOutputParser()
        answer = chain.invoke({"request": state["original_request"]})
        state["final_response"] = {"message": answer}
    except Exception as e:
        print(f"오류: 일상 대화 처리 중 오류 발생 - {e}")
        state["error"] = f"일상 대화 처리 중 오류 발생: {e}"
        state["final_response"] = {"message": "죄송합니다. 대화 처리 중 오류가 발생했습니다. 다시 시도해주세요."}
    return state

def anomalous_request_node(state: GraphState) -> GraphState:
    """이상한 요청에 대해 정중하게 거절하는 답변을 생성합니다."""
    print("--- 노드 실행: 이상한 요청 처리 ---")
    answer = (
        "죄송합니다. 요청하신 내용은 비윤리적이거나 시스템의 목적과 맞지 않아 처리할 수 없습니다. "
        "저는 알림톡 템플릿 생성, 관련 법규 안내 등 비즈니스 메시징 업무를 돕기 위해 만들어졌습니다. "
        "다른 도움이 필요하시면 말씀해주세요."
    )
    state["final_response"] = {"message": answer}
    return state

def template_generation_node(state: GraphState) -> GraphState:
    """
    템플릿 생성의 모든 단계를 처리하는 노드.
    chatbot_logic.py의 process_chat_message를 호출하여 다단계 대화를 관리합니다.
    """
    print("--- 노드 실행: 템플릿 생성 파이프라인 ---")
    try:
        # chatbot_logic의 상태(`template_pipeline_state`)와 현재 사용자 메시지(`original_request`)를 전달
        response_data = process_chat_message(
            message=state["original_request"],
            state=state["template_pipeline_state"]
        )
        # 로직 처리 후의 결과(메시지, 옵션 등)를 final_response에 저장
        state["final_response"] = response_data
        
        # 다음 단계를 위한 상태 업데이트
        updated_pipeline_state = response_data.get("state", {})
        state["template_pipeline_state"] = updated_pipeline_state
        
        # 다음 액션(프론트엔드 제어용)을 GraphState의 최상위로 끌어올림
        if "step" in updated_pipeline_state:
            state["next_action"] = updated_pipeline_state["step"]
        else:
            state["next_action"] = None 

    except Exception as e:
        print(f"오류: 템플릿 생성 중 오류 발생 - {e}")
        state["error"] = f"템플릿 생성 중 오류 발생: {e}"
        state["final_response"] = {"message": "죄송합니다. 템플릿 생성 중 오류가 발생했습니다. 다시 시도해주세요."}
        state["next_action"] = None
    return state