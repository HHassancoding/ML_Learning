#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "🚀 Starting Stock Analysis App..."

# Start the FastAPI backend in the background
echo "⚙️  Starting backend on port 8000..."
uvicorn src.stocks_full_stack.main:app --host 0.0.0.0 --port 8000 &
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

# Start the Streamlit frontend (this runs in the foreground)
echo "🎨 Starting frontend on port 8501..."
streamlit run Stocks_Full_Stack/Dashboard.py \
    --server.address=0.0.0.0 \
    --server.port=8501 \
    --browser.gatherUsageStats=false

# If Streamlit exits, the container will stop
# (which is what we want - the main process should keep running)