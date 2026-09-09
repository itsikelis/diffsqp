#!/bin/bash

# ==========================================
# Experiment Configuration
# ==========================================
N=1                 # Number of times to run each file
BATCH_SIZE=4        # Target batch size (nB)
DEVICE="cpu"        # Target device (cpu or cuda)

# The list of target scripts
SCRIPTS=(
    "cartpole/trajopt_forward.py"
    "cartpole/trajopt_inverse_lqr.py"
    "cartpole/trajopt_inverse.py"
)

# ==========================================
# Execution Loop
# ==========================================
echo "Starting experiments..."

for SCRIPT in "${SCRIPTS[@]}"; do
    # Create a safe folder name by stripping the .py extension
    BASE_NAME="${SCRIPT%.py}"

    echo "=========================================="
    echo " Processing: $SCRIPT"
    echo "=========================================="

    for (( i=1; i<=N; i++ )); do
        # Create the target directory: results/script_name/run_n
        RUN_DIR="results/${BASE_NAME}/run_$i"
        mkdir -p "$RUN_DIR"

        # Define the save file path
        SAVE_FILE="${RUN_DIR}/solution"

        echo "--- Run $i/$N ---"
        echo "Executing: python $SCRIPT -batch_size $BATCH_SIZE -device $DEVICE -save $SAVE_FILE"

        # Run the python script
        uv run "$SCRIPT" \
            -batch_size "$BATCH_SIZE" \
            -device "$DEVICE" \
            -save "$SAVE_FILE"

        # Stop the script if a run fails
        if [ $? -ne 0 ]; then
            echo "Error encountered while running $SCRIPT (Run $i). Exiting."
            exit 1
        fi
    done
    echo ""
done

echo "All runs completed and stored successfully in the 'results' directory!"
