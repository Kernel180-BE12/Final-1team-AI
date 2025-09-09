# Python 3.11 slim 버전을 기반 이미지로 사용
FROM python:3.11.9

# 작업 디렉토리를 /app으로 설정
WORKDIR /app

# requirements.txt를 먼저 복사하여 pip install 캐싱을 활용
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# [수정] 현재 디렉토리의 모든 파일과 폴더를 작업 디렉토리로 복사
# 이렇게 하면 main.py, chatbot_logic.py 뿐만 아니라 data/ 폴더도 함께 복사됩니다.
COPY . .

# uvicorn 서버 실행
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
#gunicorn 서버 실행 병렬 처리 , 비동기화
# CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--host", "0.0.0.0", "--port", "8000"]

CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "main:app"]
