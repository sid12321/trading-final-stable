#!/bin/bash
# Script to stop TensorBoard instances

echo "🛑 Stopping TensorBoard instances..."

# Find and kill TensorBoard processes
PIDS=$(pgrep -f "tensorboard")

if [ -z "$PIDS" ]; then
    echo "✅ No TensorBoard processes found"
else
    echo "📊 Stopping TensorBoard processes: $PIDS"
    kill $PIDS 2>/dev/null
    sleep 2
    
    # Force kill if still running
    REMAINING=$(pgrep -f "tensorboard")
    if [ ! -z "$REMAINING" ]; then
        echo "🔨 Force stopping remaining processes: $REMAINING"
        kill -9 $REMAINING 2>/dev/null
    fi
    
    echo "✅ TensorBoard stopped"
fi

# Check port status
for port in 6006 6007 6008; do
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "⚠️  Port $port is still in use"
    else
        echo "✅ Port $port is available"
    fi
done