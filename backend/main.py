import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import traceback
import asyncio
import json
from fastapi.responses import StreamingResponse

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
        template_state = session_state.get("template_pipeline_state")
        if not template_state:
            template_state = {'step': 'initial'}


        # LangGraph에 전달할 상태를 구성합니다.
        initial_graph_state = {
            "original_request": message,
            "template_pipeline_state":  template_state,
            "intent": None,
            "next_action": session_state.get("next_action"),
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

        # --- ▼▼▼ 여기가 수정된 부분입니다 ▼▼▼ ---
        return {
            "success": True,
            "response": response_data.get("message", "오류: 응답 메시지가 없습니다."),
            "state": new_session_state,
            "options": response_data.get("options", []),
            "template": response_data.get("template", ""),
            "structured_template": response_data.get("structured_template"),
            "editable_variables": response_data.get("editable_variables", {}),
            "structured_templates": response_data.get("structured_templates", []),
            "hasImage": response_data.get("hasImage", False) 
        }
        # --- ▲▲▲ 여기까지 수정 ▲▲▲ ---

    except Exception as e:
        print(f"API 처리 중 심각한 오류 발생: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"서버 내부 오류: {str(e)}")

@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    SSE: Content-Type text/event-stream
    Spring(WebFlux) → WebClient로 그대로 릴레이하세요.
    """
    message = request.message
    session_state = request.state or {}
    template_state = session_state.get("template_pipeline_state") or {"step": "initial"}

    initial_graph_state = {
        "original_request": message,
        "template_pipeline_state": template_state,
        "intent": None,
        "next_action": session_state.get("next_action"),
        "final_response": None,
        "retrieved_docs": None,
        "error": None
    }

    async def gen():
        # 최소한 “심장박동(heartbeat)” 한 번은 먼저 쏘기(프록시 idle 회피)
        yield f"data: 답변 생성 시작\n\n"

        # 긴 작업 동안 프록시/워커 idle 타임아웃을 피하기 위해 주기적 keepalive를 전송
        keepalive_interval_seconds = 10
        task = asyncio.create_task(
            app_graph.ainvoke(initial_graph_state, config={"recursion_limit": 10})
        )

        while not task.done():
            # 진행 중 하트비트 전송
            yield f"data: 답변 생성 중\n\n"
            await asyncio.sleep(keepalive_interval_seconds)

        final_graph_state = await task

        # 필요한 경우 중간 단계별로 쪼개서 보내세요.
        response_data = final_graph_state.get("final_response", {})
        payload = {
            "success": True,
            "response": response_data.get("message", "오류: 응답 메시지가 없습니다."),
            "state": {
                "intent": final_graph_state.get("intent"),
                "next_action": final_graph_state.get("next_action"),
                "template_pipeline_state": final_graph_state.get("template_pipeline_state", {})
            },
            "options": response_data.get("options", []),
            "template": response_data.get("template", ""),
            "structured_template": response_data.get("structured_template"),
            "editable_variables": response_data.get("editable_variables", {}),
            "structured_templates": response_data.get("structured_templates", []),
            "hasImage": response_data.get("hasImage", False)
        }
        yield f"data: {json.dumps(payload)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
