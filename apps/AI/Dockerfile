# Dockerfile for Artemis AI Backend
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (for PDF generation / Playwright)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    wkhtmltopdf \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers (for PDF service)
RUN playwright install chromium
RUN playwright install-deps chromium

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Start Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
