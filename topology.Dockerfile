FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY topology_core.py topology_api.py ./

CMD ["uvicorn", "topology_api:app", "--host", "0.0.0.0", "--port", "8000"]
