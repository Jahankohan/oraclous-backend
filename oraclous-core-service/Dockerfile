# Stage 1: Install base requirements
FROM python:3.12-slim AS extra

WORKDIR /app
COPY requirements.txt ./
RUN pip install --user --no-cache-dir -r requirements.txt

FROM python:3.12-slim

WORKDIR /app
# Use --from=extra if you want extra deps, otherwise --from=base
COPY --from=extra /root/.local /root/.local
COPY ./app /app
ENV PATH=/root/.local/bin:$PATH

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
