"""Compare solver variants on the toy problems over many perturbed starts.

    OMP_NUM_THREADS=1 uv run python tests/benchmark_robustness.py
    OMP_NUM_THREADS=1 uv run python tests/benchmark_robustness.py \
        --problems cartpole_forward --noise 1 10 --variants filter filter_ipopt

For every problem, perturbation size and variant, a batch of initial states is
solved and the script reports, per batch element:

    solved : dynamics and constraint violation < 1e-3 and the terminal state
             within 0.05 (inf-norm) of the target
    term   : the solver's own termination test fired
    iters  : SQP iterations until the solver stopped (whole batch)
    cost   : median final cost over the solved elements
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

# Overrides of the toy problem defaults for each variant
SAFEGUARDS = {
    "admm_warm_start_z": True,
    "admm_resume_kappa": 0.5,
    "ls_fail_increase_reg": False,
}
VARIANTS = {
    "filter": {"ls_function": "filter"},
    "merit": {"ls_function": "merit"},
    "filter_ipopt": {"ls_function": "filter_ipopt"},
    "merit_adaptive": {"ls_function": "merit_adaptive"},
    "merit_adaptive+z": {"ls_function": "merit_adaptive", "admm_warm_start_z": True},
    "merit_adaptive+resume": {"ls_function": "merit_adaptive", "admm_resume_kappa": 0.5},
    "merit_adaptive+noreg": {"ls_function": "merit_adaptive", "ls_fail_increase_reg": False},
    "merit_adaptive+all": {"ls_function": "merit_adaptive", **SAFEGUARDS},
    "filter_ipopt+all": {"ls_function": "filter_ipopt", **SAFEGUARDS},
    "filter+all": {"ls_function": "filter", **SAFEGUARDS},
    "merit_adaptive+soc": {"ls_function": "merit_adaptive", "second_order_correction": True},
    "merit_adaptive+resume+soc": {
        "ls_function": "merit_adaptive",
        "admm_resume_kappa": 0.5,
        "second_order_correction": True,
    },
    "filter_ipopt+soc": {"ls_function": "filter_ipopt", "second_order_correction": True},
    "filter_ipopt+resume+soc": {
        "ls_function": "filter_ipopt",
        "admm_resume_kappa": 0.5,
        "second_order_correction": True,
    },
}

TARGET_TOL = 0.05
VIOL_TOL = 1e-3


def run(problem_name, variant, batch_size, noise, seed, sqp_max_iter, dtype):
    problem, guess, args = toy_problems.PROBLEMS[problem_name](
        batch_size=batch_size, seed=seed, noise_scale=noise
    )
    args.update(VARIANTS[variant])
    args["sqp_max_iter"] = sqp_max_iter
    params = SqpParameters(**args)

    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        solution, log = sqp_solve(problem, params, guess)
    wall = time.perf_counter() - t0

    cost, dyn_inf, constr_inf, _ = problem.evaluate_guess(solution)
    x_des = problem.costs[-1][0].x_des
    target_err = torch.abs(solution.x[:, -1] - x_des).amax(dim=-1)
    viol = torch.maximum(dyn_inf, constr_inf)
    solved = (viol < VIOL_TOL) & (target_err < TARGET_TOL) & torch.isfinite(cost)

    return {
        "solved": solved.float().mean().item(),
        "term": log.envs_terminated / batch_size,
        "iters": log.sqp_iterations,
        "cost": cost[solved].median().item() if solved.any() else float("nan"),
        "viol": viol.max().item(),
        "time": wall,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", nargs="+", default=list(toy_problems.PROBLEMS))
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--noise", nargs="+", type=float, default=[1.0, 10.0, 30.0])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--sqp_max_iter", type=int, default=80)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--float64", action="store_true")
    a = parser.parse_args()

    if a.device != "cpu":
        torch.set_default_device(a.device)
    dtype = torch.float64 if a.float64 else torch.float32
    torch.set_default_dtype(dtype)

    header = f"{'problem':18s} {'noise':>5s} {'variant':26s} {'solved':>6s} {'term':>5s} {'iters':>5s} {'cost':>10s} {'max viol':>9s} {'time':>6s}"
    print(header)
    print("-" * len(header))
    for name in a.problems:
        for noise in a.noise:
            for variant in a.variants:
                rows = [
                    run(name, variant, a.batch_size, noise, seed, a.sqp_max_iter, dtype)
                    for seed in range(a.seeds)
                ]
                avg = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
                print(
                    f"{name:18s} {noise:5.0f} {variant:26s} {avg['solved']:6.0%} "
                    f"{avg['term']:5.0%} {avg['iters']:5.0f} {avg['cost']:10.4g} "
                    f"{avg['viol']:9.1e} {avg['time']:5.1f}s",
                    flush=True,
                )
        print()


if __name__ == "__main__":
    main()
