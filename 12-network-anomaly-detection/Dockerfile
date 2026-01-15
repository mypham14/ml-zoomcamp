FROM python:3.11-slim

LABEL maintainer="your-name <your-email@example.com>"

WORKDIR /app

ENV PYTHONUNBUFFERED=1

# Install system deps (keep image small)
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python deps
COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /app

# Create non-root user for better security
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && chown -R appuser:appuser /app

USER appuser

# Expose Streamlit port
EXPOSE 8501

# Entrypoint: run Streamlit in headless mode
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port", "8501", "--server.headless", "true"]