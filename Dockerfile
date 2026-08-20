FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Copy the startup script
COPY start.sh .

# Make it executable
RUN chmod +x start.sh

EXPOSE 8000 8501

# Run the startup script instead of a single command
CMD ["./start.sh"]