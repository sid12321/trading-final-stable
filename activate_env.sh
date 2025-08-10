#!/bin/bash
# Activation script for trading system virtual environment

echo "🚀 Activating trading system virtual environment..."
source venv/bin/activate

echo "📊 Environment Information:"
echo "  - Platform: macOS (Apple Silicon)"
echo "  - Python: $(python --version)"
echo "  - PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  - Device: $(python -c 'from parameters import DEVICE; print(DEVICE)')"

echo ""
echo "✅ Virtual environment activated!"
echo "💡 You can now run:"
echo "   python test_kite_login.py       # Test ChromeDriver and Kite setup"
echo "   python test_gpu_utilization.py  # Test system performance"
echo "   python model_trainer.py         # Full training pipeline (requires Kite login)"
echo "   python train_only.py            # Training-only mode (uses existing data)"
echo "   python train_only.py --help     # See training-only options"
echo "   python monitor_performance.py   # Monitor M4 Max resource utilization"
echo "   python benchmark_training.py    # Benchmark training speed"
echo "   python trader.py                # Live trading (requires Kite auth)"
echo "   ./start_tensorboard.sh           # Start TensorBoard (auto-cleans logs)"
echo "   ./stop_tensorboard.sh            # Stop TensorBoard instances"  
echo "   ./clean_tensorboard_logs.sh      # Clean TensorBoard logs manually"
echo ""
echo "🔄 To deactivate, run: deactivate"