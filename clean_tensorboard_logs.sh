#!/bin/bash
# Script to clean TensorBoard logs directory

echo "🧹 Cleaning TensorBoard logs directory..."

# Check if directory exists
if [ ! -d "tmp/tensorboard_logs" ]; then
    echo "📁 Creating tmp/tensorboard_logs/ directory"
    mkdir -p tmp/tensorboard_logs
else
    # Remove all contents inside tmp/tensorboard_logs/
    echo "🗑️  Removing existing logs from tmp/tensorboard_logs/"
    rm -rf tmp/tensorboard_logs/*
    
    # Remove hidden files like .DS_Store
    rm -f tmp/tensorboard_logs/.*
    
    echo "✅ TensorBoard logs directory cleaned"
fi

# Verify cleanup
FILES_COUNT=$(find tmp/tensorboard_logs -mindepth 1 | wc -l | tr -d ' ')
if [ "$FILES_COUNT" -eq 0 ]; then
    echo "✅ Directory is now empty and ready for new logs"
else
    echo "⚠️  Warning: $FILES_COUNT files/folders still remain"
    ls -la tmp/tensorboard_logs/
fi