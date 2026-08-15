FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project code
COPY . .

# Pre-download the embedding + reranker models into the image at build time
# instead of on the first real request -- without this, whoever hits the app
# first pays a slow cold-start download, and that request fails outright if
# Hugging Face happens to be unreachable at that exact moment.
RUN python -c "from backend.core.embeddings import get_embeddings; from backend.core.reranker import ReRanker; get_embeddings(); ReRanker().encoder; print('Models warmed.')"

# Expose Hugging Face Spaces default port
EXPOSE 7860

# Run FastAPI backend on port 7860.
# Started from /app (the project root) so `backend.*` imports resolve.
# Do NOT add --app-dir backend: that puts Python inside backend/, where there is
# no `backend` package to import, and the server fails on startup.
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
