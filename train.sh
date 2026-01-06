#!/bin/bash

# SafeRec Training Script with nohup
# Usage: ./train.sh or bash train.sh

# Create logs directory if it doesn't exist
mkdir -p logs

# Get current timestamp for log filename
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/train_${TIMESTAMP}.log"

# Optional: Set GPU device (uncomment if needed)
# export CUDA_VISIBLE_DEVICES=0

# Optional: Set other environment variables
# export PYTHONUNBUFFERED=1  # Force Python to output immediately

echo "Starting training at $(date)"
echo "Log file: ${LOG_FILE}"
echo "PID will be saved to logs/train.pid"

# Run training with nohup
# - nohup: allows process to continue after terminal closes
# - > ${LOG_FILE} 2>&1: redirect both stdout and stderr to log file
# - &: run in background
CUDA_VISIBLE_DEVICES=1 nohup /home/coder/miniconda3/envs/safe/bin/python reinforce.py > ${LOG_FILE} 2>&1 &

# Save the process ID
TRAIN_PID=$!
echo ${TRAIN_PID} > logs/train.pid

echo "Training started with PID: ${TRAIN_PID}"
echo "Monitor progress with: tail -f ${LOG_FILE}"
echo "Stop training with: kill ${TRAIN_PID}"
echo ""
echo "Useful commands:"
echo "  - Check if training is running: ps -p ${TRAIN_PID}"
echo "  - View last 50 lines of log: tail -n 50 ${LOG_FILE}"
echo "  - Follow log in real-time: tail -f ${LOG_FILE}"
