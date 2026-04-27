FROM python:3.12-slim

WORKDIR /app

RUN pip install uv

COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

COPY app.py newswatcher.py start.sh ./
COPY static/ ./static/

RUN chmod +x start.sh

EXPOSE 8420

CMD ["./start.sh"]
