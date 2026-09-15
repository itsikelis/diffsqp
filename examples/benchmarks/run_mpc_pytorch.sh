#!/bin/bash

# ==========================================
# Experiment Configuration
# ==========================================
N=5                                   # Number of times to run each batch size
BATCH_SIZES=(1 32 1024 32768)         # Array of batch sizes to test
DEVICE="cuda"                         # Target device (cpu or cuda)
GPU_ID=0                              # GPU to monitor
SCRIPT="quadrotor/mpc_pytorch.py" # The target script

# ==========================================
# Execution Loop
# ==========================================
echo "Starting batch size sweep for $SCRIPT..."

BASE_NAME="${SCRIPT%.py}"

for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
    echo "=========================================="
    echo " Processing Batch Size: $BATCH_SIZE"
    echo "=========================================="

    for (( i=1; i<=N; i++ )); do
        # Create the target directory
        RUN_DIR="results/${BASE_NAME}/batch_${BATCH_SIZE}/run_$i"
        mkdir -p "$RUN_DIR"

        # Define save paths
        SAVE_FILE="${RUN_DIR}/solution"
        VRAM_LOG="${RUN_DIR}/vram_peak.txt"

        echo "--- Run $i/$N ---"

        # 1. Start the VRAM monitor in the background and redirect output to the experiment folder
        ./track_vram.sh "$GPU_ID" > "$VRAM_LOG" &
        MONITOR_PID=$!

        # 2. Run the python script (runs in the foreground)
        echo "Executing: uv run $SCRIPT -batch_size $BATCH_SIZE -device $DEVICE -save $SAVE_FILE"

        uv run "$SCRIPT" \
            -batch_size "$BATCH_SIZE" \
            -device "$DEVICE" \
            -save "$SAVE_FILE"

        # Capture the exit code of the python script
        EXIT_CODE=$?

        # 3. Gracefully stop the VRAM monitor so it triggers its trap and saves the max mem
        kill -SIGTERM $MONITOR_PID
        wait $MONITOR_PID 2>/dev/null  # Hide the default bash termination message

        echo "Peak VRAM logged to $VRAM_LOG"

        # Stop the script if a run fails (done after killing monitor so we don't leave zombie processes)
        if [ $EXIT_CODE -ne 0 ]; then
            echo "Error encountered while running $SCRIPT with batch size $BATCH_SIZE (Run $i). Exiting."
            exit 1
        fi
    done
    echo ""
done

echo "All runs completed and stored successfully in the 'results' directory!"
