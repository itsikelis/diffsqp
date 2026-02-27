#! /bin/bash

# Number of executions for each experiment
N_RUNS=10
STD=0.001
DEV=cuda

dynamics="forward inverse"
batch_sizes="1 8 32 128 512 2048 8192"

# Save current directory path
cwd=$(pwd)

mkdir -p results/
cd results/


mkdir -p sqp_comparison/
cd sqp_comparison/

# Run diffmpc
mkdir -p diffmpc_results/
cd diffmpc_results/
for size in $batch_sizes; do
    mkdir -p "${size}_envs"/
    cd "${size}_envs"/
    for i in $(seq 1 $N_RUNS); do
        mkdir -p "run_$i"/
        cd "run_$i"/
        rm -rf *
        uv run "${cwd}/cartpole_diffmpc.py" -nb ${size} -std ${STD}
        cd ../
    done
    cd ../
done
cd ../

# Run ssqp
mkdir -p ssqp_results/
cd ssqp_results/
for dyn in $dynamics; do
    mkdir -p "${dyn}_dynamics"/
    cd "${dyn}_dynamics"/
    for size in $batch_sizes; do
        mkdir -p "${size}_envs"/
        cd "${size}_envs"/
        for i in $(seq 1 $N_RUNS); do
            mkdir -p "run_$i"/
            cd "run_$i"/
            rm -rf *
            uv run "${cwd}/cartpole_sqp.py" -nb ${size} -dev ${DEV} -model ${dyn} -std ${STD}
            cd ../
        done
        cd ../
    done
    cd ../
done
cd ../
