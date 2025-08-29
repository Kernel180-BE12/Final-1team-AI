# /backend/Dockerfile (변경 없음)
FROM python:3.11.9
WORKDIR /app
COPY ./requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app/
ARG OPENAI_API_KEY
ENV OPENAI_API_KEY=${OPENAI_API_KEY}
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0..0.0.0", "--port", "8000"]