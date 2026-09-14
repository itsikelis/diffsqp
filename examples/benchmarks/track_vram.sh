#!/usr/bin/env bash

# Default GPU index to monitor
GPU_ID=${1:-0}
POLL_INTERVAL=0.1 # Seconds between checks

max_mem=0

# Trap Ctrl+C (SIGINT) and normal exit (SIGTERM, EXIT) to print peak memory
cleanup() {
    trap - EXIT
    echo "${max_mem}"
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

while true; do
    # Query current memory usage in MiB
    current_mem=$(nvidia-smi --id="$GPU_ID" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)

    # Check if value is a valid integer and update max
    if [[ "$current_mem" =~ ^[0-9]+$ ]]; then
        if (( current_mem > max_mem )); then
            max_mem=$current_mem
        fi
    fi

    sleep "$POLL_INTERVAL"
done
