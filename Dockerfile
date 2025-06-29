# OrgGPT Dockerfile for Hugging Face Spaces deployment
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js for frontend build
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs

# Copy backend requirements and install Python dependencies
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip3 install --no-cache-dir -r backend/requirements.txt

# Copy backend code
COPY backend/ /app/backend/

# Copy frontend code and build
COPY frontend/ /app/frontend/
WORKDIR /app/frontend

# Install frontend dependencies and build
RUN npm install -g pnpm
RUN pnpm install
RUN pnpm run build

# Copy built frontend to backend static directory
RUN mkdir -p /app/backend/static
RUN cp -r /app/frontend/dist/* /app/backend/static/

# Switch back to app directory
WORKDIR /app

# Copy environment and configuration files
COPY .env.production /app/.env
COPY README.md /app/
COPY LICENSE /app/

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/api/v1/health || exit 1

# Run the application
CMD ["python3", "backend/main.py"]
