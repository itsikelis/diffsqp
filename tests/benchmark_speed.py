"""Time a fixed amount of solver work on the toy problems.

    OMP_NUM_THREADS=1 uv run python tests/benchmark_speed.py
    OMP_NUM_THREADS=1 uv run python tests/benchmark_speed.py --devices cuda --batch_sizes 4 1024

Every run does `--sqp_iters` SQP iterations with exactly `--admm_iters` ADMM
iterations each (tolerances disabled), so the numbers compare implementations
and settings, not convergence. Reported: milliseconds per ADMM iteration
(total time / total ADMM iterations, so it includes linearization and line
search) and the total time.

On CPU, `torch.set_default_device` is not called: its dispatch overhead adds
~70% to small-batch runs, and CPU is the default anyway.
"""

import argparse
import contextlib
import io
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
import toy_problems  # noqa: E402

from diffsqp.solvers import sqp_solve, SqpParameters  # noqa: E402


def time_run(name, batch_size, device, sqp_iters, admm_iters, parallel_scan):
    problem, guess, args = toy_problems.PROBLEMS[name](batch_size=batch_size)
    args.update(
        sqp_max_iter=sqp_iters,
        sqp_cost_eps=-1.0,
        admm_max_iter=admm_iters,
        admm_abs_tolerance=-1.0,
        admm_rel_tolerance=-1.0,
        lqr_parallel_scan=parallel_scan,
        verbose=False,
    )
    params = SqpParameters(**args)

    def run():
        with contextlib.redirect_stdout(io.StringIO()):
            return sqp_solve(problem, params, guess)

    if device != "cpu":
        run()  # warm up CUDA context and allocator
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    _, log = run()
    if device != "cpu":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return elapsed, sum(log.admm_iter_hist)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", nargs="+", default=list(toy_problems.PROBLEMS))
    parser.add_argument("--devices", nargs="+", default=["cpu"])
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[4, 256])
    parser.add_argument("--sqp_iters", type=int, default=2)
    parser.add_argument("--admm_iters", type=int, default=30)
    parser.add_argument("--parallel_scan", action="store_true")
    a = parser.parse_args()

    print(f"{'problem':18s} {'device':6s} {'batch':>5s} {'ms/ADMM it':>10s} {'total':>8s}")
    for device in a.devices:
        torch.set_default_device(None if device == "cpu" else device)
        for name in a.problems:
            for batch_size in a.batch_sizes:
                elapsed, admm_total = time_run(
                    name, batch_size, device, a.sqp_iters, a.admm_iters, a.parallel_scan
                )
                print(
                    f"{name:18s} {device:6s} {batch_size:5d} "
                    f"{elapsed / admm_total * 1e3:10.2f} {elapsed:7.2f}s",
                    flush=True,
                )


if __name__ == "__main__":
    main()
