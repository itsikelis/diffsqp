#! /bin/bash

# Number of executions for each experiment
N_RUNS=5
STD=0.1

devices="cpu cuda"
dynamics="forward inverse"
batch_sizes="1 8 32 128 512 2048 8192"

# Save current directory path
cwd=$(pwd)

mkdir -p results/
cd results/

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
                for i in $(seq 1 $N_RUNS); do
                    mkdir -p "run_$i"/
                    cd "run_$i"/
                    rm -rf *
                    # echo $(pwd)
                    # echo $cwd
                    uv run "${cwd}/paper_experiments/cartpole_experiment.py" -nb ${size} -model ${dyn} -qp lqr -dev ${dev} -task swingup -std ${STD}
                    clear
                    cd ../
                done
            cd ../
        done
        cd ../
    done
    cd ../
done

# mkdir -p cpu_parallel/
# cd cpu_parallel
#
# mkdir -p forward_dyn/
# cd forward_dyn
#
# for i in $(seq 1 $N_RUNS); do
#     mkdir -p "run_$i"/
#     cd "run_$i"/
#     rm -rf *
#
#     uv run $cwd/cartpole_experiment.py -nb 8 -model inverse -qp lqr -dev cpu -task balance -std 0.0
#
#     cd ../
# done
# cd ../ # cd to cpu_parallel/
