# TensorBoard Usage Guide

## Starting TensorBoard

```bash
# Start TensorBoard (automatically cleans logs and finds available port)
./start_tensorboard.sh

# Stop all TensorBoard instances
./stop_tensorboard.sh

# Clean logs manually (optional - done automatically)
./clean_tensorboard_logs.sh
```

## Accessing TensorBoard

- **Default URL**: http://localhost:6006
- **If port 6006 is busy**: The script will automatically try 6007, 6008, etc.
- **Current Logs**: `tmp/tensorboard_logs/`

## Available Training Logs

Current logs in the system:
- `BPCL_final_20250809_002752/` - Recent BPCL training session
- Training metrics and model performance data available

## Monitoring Training

**Automatic Cleanup**: TensorBoard logs are automatically cleaned before each training run for fresh metrics.

When you run `python model_trainer.py`, training metrics will be logged to:
- `tmp/tensorboard_logs/[SYMBOL]_final_[TIMESTAMP]/`
- Metrics include: loss, rewards, policy gradients, value functions
- Previous logs are automatically removed to keep data fresh

## TensorBoard Features

- **Scalars**: Training loss, rewards, policy metrics
- **Histograms**: Weight distributions, gradient norms
- **Images**: Model architecture graphs (if enabled)
- **Time Series**: Training progress over episodes

## Tips

1. **Start TensorBoard before training** to see real-time updates
2. **Refresh browser** if metrics don't update immediately
3. **Use different ports** if running multiple experiments
4. **Archive old logs** to keep the interface clean

## Troubleshooting

- **Port in use**: Script automatically finds next available port
- **No data**: Check that training is writing to `tmp/tensorboard_logs/`
- **Browser issues**: Try clearing cache or different browser
- **Permission errors**: Ensure `tmp/` directory is writable