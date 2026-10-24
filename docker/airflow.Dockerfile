# Airflow 3 with the project and its packages.
# Built by docker-compose.yaml at the repository root.
FROM apache/airflow:3.3.2-python3.12

USER airflow
# CPU-only PyTorch first (much smaller than the default build), for the GNN tasks
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# The project's packages. Naming apache-airflow again keeps pip from changing its version.
COPY --chown=airflow:root requirements.txt requirements-pipeline.txt /opt/ecowas/
RUN pip install --no-cache-dir "apache-airflow==3.3.2" -r /opt/ecowas/requirements.txt

# The project itself. Airflow reads the DAGs from /opt/ecowas/dags.
COPY --chown=airflow:root . /opt/ecowas
ENV PYTHONPATH=/opt/ecowas
WORKDIR /opt/ecowas
