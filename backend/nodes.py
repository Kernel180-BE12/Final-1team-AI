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
llm_fast = ChatOpenAI(
            model="gpt-4o-mini",temperature=0.5
        )
llm_smart = ChatOpenAI(
            model="gpt-4o",
            temperature=0.2
        )

# llm_smart = ChatOpenAI(model="gpt-5", temperature=0.2)  # gpt-5로 변경 (가정: OpenAI에서 지원될 경우)
# llm_fast = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # nano를 gpt-4o-mini로 해석 (nano가 gpt-4o-mini를 의미할 가능성이 높음)

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

def route_request_node(state: GraphState) -> GraphState:
    """그래프의 진입점으로, 상태를 변경하지 않고 라우팅 엣지의 시작점 역할을 합니다."""
    print("--- 메인 라우터 진입 ---")
    return state

def classify_intent_node(state: GraphState) -> GraphState:
    """사용자의 새로운 요청 의도를 분류합니다. (개선된 프롬프트)"""
    print("--- 노드 실행: 의도 분류 ---")

    # --- 수정된 프롬프트 ---
    system_prompt = """당신은 비즈니스 메시징 솔루션 'Jober'의 전문 AI 어시스턴트입니다. 당신의 주요 임무는 사용자의 요청을 분석하여 다음 네 가지 의도 중 하나로 분류하는 것입니다. 아래의 가이드라인과 예시를 참고하여 가장 적절한 의도를 JSON 형식으로만 출력해주세요.

[의도 분류 가이드라인]
1.  **template_generation (템플릿 생성)**: 사용자가 메시지 생성을 명시적으로 요청하거나, 특정 내용/초안/예시를 제공하며 템플릿 생성을 암시하는 경우.
2.  **legal_inquiry (법률 문의)**: 정보통신망법, 알림톡 가이드라인 등 법률, 규제에 대해 질문하는 경우.
3.  **chit_chat (일상 대화)**: 업무와 관련 없는 일반적인 대화, 인사, 안부. 단, 메시지 예시를 포함한 인사는 'template_generation'으로 분류.
4.  **anomalous_request (이상 요청)**: 비윤리적이거나 시스템 목적과 무관한 부적절한 요청.

[의도 분류 예시]
# --- Template Generation Examples ---
- 사용자: "병원 예약 하루 전날 보내는 리마인드 메시지 좀" -> {{ "intent": "template_generation" }}
- 사용자: "결제 완료되면 나가는 메시지 초안" -> {{ "intent": "template_generation" }}
- 사용자: (별다른 설명 없이 내용만 입력)
    [Web발신]
    고객님, 주문하신 상품의 배송이 시작되었습니다.
    운송장번호: 123456789
    -> {{ "intent": "template_generation" }}
- 사용자: "이걸로 알림톡 보내려고요. 안녕하세요. 마케팅리즈입니다. 9월 신규 고객 대상 할인 쿠폰이 발급되었습니다." -> {{ "intent": "template_generation" }}

# --- Legal Inquiry Examples ---
- 사용자: "광고 메시지는 밤 9시 이후에 보내면 안된다고 들었는데 맞나요?" -> {{ "intent": "legal_inquiry" }}
- 사용자: "이벤트 안내 메시지 보낼 때 수신거부 방법도 꼭 넣어야 하는지 궁금합니다." -> {{ "intent": "legal_inquiry" }}
- 사용자: "회원가입만 한 고객에게 마케팅 메시지를 보내도 법적으로 문제가 없나요?" -> {{ "intent": "legal_inquiry" }}

# --- Chit-chat Examples ---
- 사용자: "고마워요!" -> {{ "intent": "chit_chat" }}
- 사용자: "Jober 똑똑한데?" -> {{ "intent": "chit_chat" }}
- 사용자: "마케팅 특강 공지를 해야 하는데..." -> {{ "intent": "chit_chat" }}

# --- Anomalous Request Examples ---
- 사용자: "경쟁사 서비스가 안 좋다는 내용의 메시지를 대량으로 보내줘." -> {{ "intent": "anomalous_request" }}
- 사용자: "오늘 저녁 메뉴 추천해줘." -> {{ "intent": "anomalous_request" }}

이제 아래 사용자 메시지를 분석하여 의도를 분류해주세요."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "사용자 메시지: {request}")
    ])
    
    classifier_chain = prompt | llm_fast.with_structured_output(IntentClassifier)
    
    try:
        result = classifier_chain.invoke({"request": state["original_request"]})
        state["intent"] = result.intent
        print(f"-> 분류된 의도: {result.intent}")
    except Exception as e:
        print(f"오류: 의도 분류 실패 - {e}")
        state["error"] = f"의도 분류 중 오류 발생: {e}"
        state["intent"] = "chit_chat"
        state["final_response"] = {"message": "죄송합니다. 요청을 이해하는 중 문제가 발생했습니다. 다른 방식으로 질문해주시겠어요?"}
    return state



def template_confirmation_node(state: GraphState) -> GraphState:
    """템플릿 생성 의도일 경우, 사용자에게 진행 여부를 확인하고 원래 요청을 저장합니다."""
    print("--- 노드 실행: 템플릿 생성 확인 ---")
    pipeline_state = state.get("template_pipeline_state", {'step': 'initial'})
    pipeline_state['original_request'] = state['original_request']
    state['template_pipeline_state'] = pipeline_state
    message = "템플릿 생성을 요청하셨네요. 계속 진행할까요?"
    options = ["예", "아니오"]
    state["final_response"] = {"message": message, "options": options}
    state["next_action"] = "awaiting_confirmation"
    return state

def cancel_node(state: GraphState) -> GraphState:
    """프로세스 취소 시 메시지를 출력하고 상태를 초기화합니다."""
    print("--- 노드 실행: 생성 취소 ---")
    message = "알겠습니다. 템플릿 생성을 취소했습니다. 다른 도움이 필요하시면 말씀해주세요."
    state["template_pipeline_state"] = {'step': 'initial'}
    state["next_action"] = None
    state["final_response"] = {"message": message, "options": []}
    return state

def legal_inquiry_node(state: GraphState) -> GraphState:
    """법률/가이드라인 정보를 검색하고 답변을 생성합니다. (compliance 및 generation 리트리버 모두 활용)"""
    print("--- 노드 실행: 법률/가이드라인 안내 ---")
    query = state["original_request"]

    # 사용 가능한 리트리버 확인
    compliance_retriever = retrievers.get('compliance')
    generation_retriever = retrievers.get('generation')

    if not compliance_retriever and not generation_retriever:
        state["final_response"] = {"message": "죄송합니다. 법률 또는 생성 가이드라인 정보 데이터베이스가 현재 준비되지 않아 답변할 수 없습니다."}
        state["error"] = "Compliance and generation retrievers not initialized or found."
        return state

    try:
        all_docs = []
        # 'compliance' 리트리버에서 문서 검색
        if compliance_retriever:
            print("-> compliance 리트리버에서 정보 검색 중...")
            docs_compliance = compliance_retriever.invoke(query)
            all_docs.extend(docs_compliance)
            print(f"-> compliance 문서 {len(docs_compliance)}개 검색 완료.")

        # 'generation' 리트리버에서 문서 검색
        if generation_retriever:
            print("-> generation 리트리버에서 정보 검색 중...")
            docs_generation = generation_retriever.invoke(query)
            all_docs.extend(docs_generation)
            print(f"-> generation 문서 {len(docs_generation)}개 검색 완료.")

        # 검색된 문서들의 중복 제거 (내용 기준)
        unique_docs = list({doc.page_content: doc for doc in all_docs}.values())
        print(f"-> 총 {len(all_docs)}개 문서 검색, 중복 제거 후 {len(unique_docs)}개 문서 확보.")

        context = "\n\n".join([f"문서 {i+1}:\n{doc.page_content}" for i, doc in enumerate(unique_docs)])
        state["retrieved_docs"] = [doc.page_content for doc in unique_docs]

        # 프롬프트 수정: 더 포괄적인 전문가 역할 부여
        prompt = ChatPromptTemplate.from_template(
            """당신은 정보통신망법, 알림톡 가이드라인, 효과적인 메시지 작성법 등 비즈니스 메시징 규정 및 가이드라인 전문가입니다.
            아래에 제공된 '관련 규정 및 가이드'만을 근거로 사용자의 질문에 대해 정확하고 명료하게 답변해주세요.
            답변 시, 어떤 자료를 근거로 하였는지 간략히 언급해주세요. 추측성 답변은 절대 금지입니다.
            만약 관련 정보가 없다면, 정보가 없다고 솔직하게 답변해주세요.

            [관련 규정 및 가이드]
            {context}

            [사용자 질문]
            {query}

            [전문가 답변]"""
        )
        
        response_chain = prompt | llm_smart | StrOutputParser()
        answer = response_chain.invoke({"context": context, "query": query})
        state["final_response"] = {"message": answer}

    except Exception as e:
        print(f"오류: 법률/가이드라인 안내 중 오류 발생 - {e}")
        state["error"] = f"법률/가이드라인 안내 중 오류 발생: {e}"
        state["final_response"] = {"message": "죄송합니다. 관련 정보를 처리하는 중 오류가 발생했습니다."}
    
    return state

def chit_chat_node(state: GraphState) -> GraphState:
    """일상 대화에 대한 답변을 생성합니다."""
    print("--- 노드 실행: 일상 대화 ---")
    prompt = ChatPromptTemplate.from_template(
        "당신은 사용자의 비즈니스 메시징 업무를 돕는 친절하고 전문적인 AI 어시스턴트 'Jober'입니다. 사용자의 대화에 대해 자연스럽고 긍정적으로 답변하고 템플릿 생성을 요청을 이끌어내는 대답을 해주세요.만약 사용자가 너무 일상적인 대화를 한다면 비지니스 챗봇이라는걸 강조하고 템플릿 생성을 요청하는 대답을 해주세요. \n사용자 메시지: {request}"
    )
    try:
        chain = prompt | llm_fast | StrOutputParser()
        answer = chain.invoke({"request": state["original_request"]})
        state["final_response"] = {"message": answer}
    except Exception as e:
        print(f"오류: 일상 대화 처리 중 오류 발생 - {e}")
        state["error"] = f"일상 대화 처리 중 오류 발생: {e}"
        state["final_response"] = {"message": "죄송합니다. 대화 처리 중 오류가 발생했습니다."}
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
    """템플릿 생성의 모든 단계를 처리하는 노드."""
    print("--- 노드 실행: 템플릿 생성 파이프라인 ---")
    try:
        response_data = process_chat_message(
            message=state["original_request"],
            state=state["template_pipeline_state"]
        )
        state["final_response"] = response_data
        updated_pipeline_state = response_data.get("state", {})
        state["template_pipeline_state"] = updated_pipeline_state
        if "step" in updated_pipeline_state:
            state["next_action"] = updated_pipeline_state["step"]
        else:
            state["next_action"] = None
    except Exception as e:
        print(f"오류: 템플릿 생성 중 오류 발생 - {e}")
        state["error"] = f"템플릿 생성 중 오류 발생: {e}"
        state["final_response"] = {"message": "죄송합니다. 템플릿 생성 중 오류가 발생했습니다."}
        state["next_action"] = None
    return state