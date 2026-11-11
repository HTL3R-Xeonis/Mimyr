FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-deu tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY attachment_inputs.py topology_core.py topology_api.py ./

CMD ["uvicorn", "topology_api:app", "--host", "0.0.0.0", "--port", "8000"]
