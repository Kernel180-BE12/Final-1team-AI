# Python 3.11 slim 버전을 기반 이미지로 사용
FROM python:3.11.9

# 작업 디렉토리를 /app으로 설정
WORKDIR /app

# requirements.txt를 먼저 복사하여 pip install 캐싱을 활용
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY . .

CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "main:app"]
