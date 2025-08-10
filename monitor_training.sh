#!/bin/bash
# Monitor training progress

echo "Monitoring training progress..."
echo "================================"

# Check for tensorboard logs
if [ -d "tmp/tensorboard_logs" ]; then
    echo "TensorBoard logs available at: tmp/tensorboard_logs"
    echo "Run 'tensorboard --logdir=tmp/tensorboard_logs' to view"
fi

# Check for checkpoints
if [ -d "tmp/checkpoints" ]; then
    echo ""
    echo "Latest checkpoints:"
    ls -lht tmp/checkpoints/ | head -5
fi

# Check for metrics
if [ -f "tmp/sb3_log/custom_metrics.txt" ]; then
    echo ""
    echo "Latest training metrics:"
    tail -10 tmp/sb3_log/custom_metrics.txt
fi

# Check for model files
echo ""
echo "Existing models:"
ls -lh models/*.zip 2>/dev/null || echo "No models found yet"

echo ""
echo "Training is running in the background..."