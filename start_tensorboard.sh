#!/bin/bash
# Script to start TensorBoard for trading system monitoring

echo "🚀 Starting TensorBoard for trading system monitoring..."

# Clean TensorBoard logs before starting
./clean_tensorboard_logs.sh
echo ""

# Activate virtual environment
source venv/bin/activate

# Function to find available port
find_available_port() {
    local port=6006
    while lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; do
        echo "⚠️  Port $port is in use, trying $((port+1))..."
        ((port++))
    done
    echo $port
}

# Find an available port
PORT=$(find_available_port)

# Start TensorBoard
echo "📊 TensorBoard will be available at: http://localhost:$PORT"
echo "📁 Monitoring directory: tmp/tensorboard_logs/"
echo ""
echo "Press Ctrl+C to stop TensorBoard"
echo "=================================================="

tensorboard --logdir=tmp/tensorboard_logs/ --port=$PORT --host=localhost