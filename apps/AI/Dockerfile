# Trigger build
FROM python:3.11-slim

# Install system dependencies for WeasyPrint, fonts, and general build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    python3-cffi \
    libcairo2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libpangoft2-1.0-0 \
    libharfbuzz0b \
    libgdk-pixbuf-2.0-0 \
    libffi-dev \
    shared-mime-info \
    fonts-liberation \
    fontconfig \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy python requirements list
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set playwright browser path environment variable and library path
ENV PLAYWRIGHT_BROWSERS_PATH=/app/.cache/ms-playwright
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/lib

# Copy the rest of the application files
COPY . .

# Install Playwright Chromium and its system dependencies
RUN apt-get update && playwright install --with-deps chromium && rm -rf /var/lib/apt/lists/*

# Run the FastAPI app with Uvicorn, dynamically binding to the port specified by Railway
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
