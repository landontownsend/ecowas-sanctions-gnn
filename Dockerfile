# Reproducible environment for the whole project.
#   docker build -t ecowas-gnn .
#   docker run --rm ecowas-gnn                      # full run with the saved data
#   docker run --rm ecowas-gnn ./run_all.sh --live  # full run with live World Bank data
FROM python:3.11-slim

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# CPU-only PyTorch first: much smaller than the default build with GPU libraries
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt requirements-pipeline.txt ./
RUN pip install -r requirements.txt

COPY . .

CMD ["./run_all.sh"]
