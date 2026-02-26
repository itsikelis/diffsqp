#! /bin/bash

# Number of executions for each experiment
N_RUNS=10
STD=0.05

devices="cpu cuda"
dynamics="forward inverse"
batch_sizes="1 8 32 128 512 2048 8192"

# Save current directory path
cwd=$(pwd)

mkdir -p results/
cd results/

# Compare CPU vs GPU parallel execution for swingup task
mkdir -p execution_time/
cd execution_time
for dev in $devices; do
    mkdir -p "${dev}_parallel/"
    cd "${dev}_parallel/"
    for dyn in $dynamics; do
        mkdir -p "${dyn}_dynamics"/
        cd "${dyn}_dynamics"/
        for size in $batch_sizes; do
            mkdir -p "${size}_envs"/
            cd "${size}_envs"/
            rm -rf *
            uv run "${cwd}/cartpole_experiment.py" -nb ${size} -model ${dyn} -qp lqr -dev ${dev} -task swingup -std ${STD}
            cd ../
        done
        cd ../
    done
    cd ../
done
cd ../ # cd to results
