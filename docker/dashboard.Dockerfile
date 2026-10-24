# The Streamlit dashboard. Built by docker-compose.yaml at the repository root.
FROM python:3.12-slim
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY dashboard/requirements.txt dashboard/requirements.txt
RUN pip install -r dashboard/requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
