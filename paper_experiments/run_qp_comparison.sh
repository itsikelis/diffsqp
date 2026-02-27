#! /bin/bash

# Number of executions for each experiment
N_RUNS=10
N_BTCH=16
STD=0.01
DEV="cuda"
DYN="inverse"

solvers="lqr kkt qpth"
tasks="swingup"

# Save current directory path
cwd=$(pwd)

mkdir -p results/
cd results/

# Compare different implementations of the internal qp solver
mkdir -p qp_comparison/
cd qp_comparison

for solver in $solvers; do
    mkdir -p "${solver}/"
    cd "${solver}/"
    for task in $tasks; do
        mkdir -p "${task}_task"/
        cd "${task}_task"/
        for i in $(seq 1 $N_RUNS); do
            mkdir -p "run_$i"/
            cd "run_$i"/
            rm -rf *
            uv run "${cwd}/cartpole_experiment.py" -nb ${N_BTCH} -model ${DYN} -qp ${solver} -dev ${DEV} -task ${task} -std ${STD}
            cd ../
        done
        cd ../
    done
    cd ../
done

cd ../ # cd to results
