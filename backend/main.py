import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import traceback

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

load_dotenv()

from graph import app_graph
from chatbot_logic import initialize_system

# --- FastAPI 앱 생성 및 설정 ---
app = FastAPI()

# CORS 설정
origins = ["*"] # 개발 중에는 모든 출처 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 요청/응답 모델
class ChatRequest(BaseModel):
    message: str
    state: Optional[Dict[str, Any]] = Field(default_factory=dict)

# 서버 시작 시 초기화
@app.on_event("startup")
async def startup_event():
    print("서버 시작: 시스템 초기화를 진행합니다...")
    initialize_system()
    print("시스템 초기화 완료.")

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """LangGraph 기반으로 통합된 챗봇 엔드포인트."""
    try:
        message = request.message
        session_state = request.state or {}

        # LangGraph에 전달할 상태를 구성합니다.
        initial_graph_state = {
            "original_request": message,
            "template_pipeline_state": session_state.get("template_pipeline_state", {
                'step': 'initial'
            }),
            "intent": None,
            # ▼▼▼ 바로 이 부분 문제입니다! None으로 초기화하는 대신 세션에서 값을 가져오도록 수정합니다. ▼▼▼
            "next_action": session_state.get("next_action"), # <--- 수정된 부분
            "final_response": None,
            "retrieved_docs": None,
            "error": None
        }
        
        print(f"새로운 요청 수신: '{message}' -> LangGraph 실행")
        
        # LangGraph 워크플로우 실행
        final_graph_state = app_graph.invoke(initial_graph_state, config={"recursion_limit": 10})
        
        # 그래프 실행 결과에서 최종 응답 데이터 추출
        response_data = final_graph_state.get("final_response", {})
        
        # 프론트엔드에 전달할 다음 대화의 세션 상태 구성
        new_session_state = {
            "intent": final_graph_state.get("intent"),
            "next_action": final_graph_state.get("next_action"),
            "template_pipeline_state": final_graph_state.get("template_pipeline_state", {})
        }

        return {
            "success": True,
            "response": response_data.get("message", "오류: 응답 메시지가 없습니다."),
            "state": new_session_state,
            "options": response_data.get("options", []),
            "template": response_data.get("template", ""),
            "structured_template": response_data.get("structured_template"),
            "editable_variables": response_data.get("editable_variables", {}),
            "structured_templates": response_data.get("structured_templates", []),
        }

    except Exception as e:
        print(f"API 처리 중 심각한 오류 발생: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"서버 내부 오류: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)