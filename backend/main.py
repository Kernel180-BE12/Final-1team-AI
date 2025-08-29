# src/main.py (FastAPI 버전으로 수정)

import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict
from dotenv import load_dotenv
import traceback

load_dotenv()

# DON'T CHANGE THIS !!!
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# 1단계에서 이름을 변경한 챗봇 로직 파일을 임포트합니다.
from backend.chatbot_logic import initialize_system, process_chat_message

# --- FastAPI 앱 생성 및 설정 ---
app = FastAPI()

# --- CORS 미들웨어 추가 (프론트엔드와 통신 허용) ---
# 프론트엔드 개발 서버의 주소 목록
origins = [
    "http://localhost",
    "http://localhost:5174", # Vite React 개발 서버의 기본 주소
    "http://127.0.0.1:5174",
    "http://localhost:3000", # React 개발 서버 추가
    "http://127.0.0.1:3000",
    # 필요하다면 프론트엔드 담당자의 다른 주소도 추가할 수 있습니다.
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API 요청/응답 모델 정의 ---
class ChatRequest(BaseModel):
    message: str
    state: Optional[Dict] = Field(default_factory=dict)

# --- FastAPI 이벤트 핸들러 ---
@app.on_event("startup")
async def startup_event():
    """
    FastAPI 서버가 시작될 때 단 한 번만 실행됩니다.
    무거운 모델 로딩 및 초기화 작업을 여기서 수행합니다.
    """
    print("서버 시작: 시스템 초기화를 진행합니다...")
    try:
        initialize_system()
        print("시스템 초기화 완료.")
    except Exception as e:
        print(f"시스템 초기화 실패: {e}")
        traceback.print_exc()

# --- API 엔드포인트 정의 ---
@app.post("/api/chat")
async def chat(request: ChatRequest) -> Dict:
    """
    챗봇 메시지를 처리하는 메인 엔드포인트입니다.
    """
    try:
        session_state = request.state
        # 세션 상태가 비어있을 경우 초기화
        if not session_state:
            session_state = {
                'step': 'initial',
                'original_request': '',
                'user_choice': '',
                'selected_style': '',
                'template_draft': '',
                'validation_result': None,
                'correction_attempts': 0
            }
        
        response_data = process_chat_message(request.message, session_state)
        
        # 프론트엔드가 필요로 하는 모든 키를 포함하여 응답을 구성합니다.
        return {
            'success': True,
            'response': response_data.get('message', ''),
            'state': response_data.get('state', {}),
            'options': response_data.get('options', []),
            'template': response_data.get('template', ''),
            'html_preview': response_data.get('html_preview', ''),
            'editable_variables': response_data.get('editable_variables', {}),
            'step': response_data.get('state', {}).get('step', 'initial')
        }
    except Exception as e:
        print(f"API 처리 중 오류 발생: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"서버 내부 오류: {str(e)}")

@app.get("/api/health")
async def health_check():
    """
    서버가 정상적으로 작동하는지 확인하는 간단한 헬스 체크 엔드포인트입니다.
    """
    return {"status": "healthy"}

# --- 정적 파일 서빙 (기존 Flask의 serve 함수 대체) ---
# 이 부분은 실제 배포 시 Nginx 같은 웹서버가 담당하는 것이 더 효율적이지만,
# 로컬 테스트를 위해 FastAPI로 간단히 구현할 수 있습니다.
# 하지만 현재 프론트엔드 개발 서버를 따로 실행하는 방식에서는 이 부분이 없어도 무방합니다.
# 만약 필요하다면 아래 주석을 해제하고 경로를 맞춰주세요.
# from fastapi.staticfiles import StaticFiles
# app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static"), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

