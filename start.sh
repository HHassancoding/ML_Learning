#!/bin/bash

set -e

echo "🚀 Starting Stock Analysis App..."

# Add current directory to Python path
export PYTHONPATH=/app:$PYTHONPATH

# Start the FastAPI backend in the background (not exposed publicly)
echo "⚙️  Starting backend on port 8000..."
uvicorn src.stocks_full_stack.main:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

echo "✅ Backend started (PID: $BACKEND_PID)"

# Wait for backend to be ready
echo "⏳ Waiting for backend to initialize..."
sleep 5

# Check if backend is running
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "❌ Backend failed to start!"
    exit 1
fi

echo "✅ Backend is running"

# Start the Streamlit frontend (this is the main process - exposed publicly)
echo "🎨 Starting frontend on port 8501..."
cd /app/src/stocks_full_stack
streamlit run Dashboard.py --server.address=0.0.0.0 --server.port=8501 --browser.gatherUsageStats=false