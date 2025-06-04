FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      python3-dev \
      libblas-dev \
      liblapack-dev \
      && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . ./
CMD ["sh"]
