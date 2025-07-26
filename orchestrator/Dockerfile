FROM python:3.12-slim

WORKDIR /app

# Install system packages required by some MCP SDKs (e.g., if future dependencies need them)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY ./ ./

# Install Python dependencies
RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
