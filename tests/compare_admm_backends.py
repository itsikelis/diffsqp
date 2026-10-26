"""Torch vs bsqp ADMM on the examples: wall time and solution matching.

Runs each example's `main` with `sqp_solve` replaced by a wrapper that solves
the same problem with both backends (same initial guess, same seed), and
reports the SQP wall time, the time spent in the QP (ADMM) calls and the
difference between the two solutions.

    OMP_NUM_THREADS=1 uv run python tests/compare_admm_backends.py
    uv run python tests/compare_admm_backends.py --devices cuda --batch_size 256
    # Same rho for both backends: the SQP iterates should match closely
    uv run python tests/compare_admm_backends.py --set admm_update_rho=False
"""

import ast

import argparse
import contextlib
import copy
import importlib.util
import io
import os
import sys
import time
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

import diffsqp.solvers.sqp as sqp  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = [
    "double_integrator",
    "cartpole_forward",
    "cartpole_inverse",
    "franka_constrained",
    "franka_se3_tracking",
    "quadrotor_se3_trajopt",
]
BACKENDS = ["torch", "bsqp"]


class QpTimer:
    """Accumulated time of the QP calls of `sqp_solve` (synchronized)."""

    def __init__(self, device):
        self.device = device
        self.seconds = 0.0
        self.calls = 0

    def wrap(self, fn):
        def timed(*args, **kwargs):
            self._sync()
            t0 = time.perf_counter()
            out = fn(*args, **kwargs)
            self._sync()
            self.seconds += time.perf_counter() - t0
            self.calls += 1
            return out

        return timed

    def _sync(self):
        if self.device == "cuda":
            torch.cuda.synchronize()


def solve_with(backend, problem, parameters, guess, device, repeats):
    """Best of `repeats` solves (the first one also warms up)."""
    names = {
        "torch": ["admm_qp_solve", "admm_qp_resume"],
        "bsqp": ["bsqp_qp_solve", "bsqp_qp_resume"],
    }[backend]
    params = copy.copy(parameters)
    params.admm_backend = backend
    params.verbose = False
    best = None
    for _ in range(repeats):
        timer = QpTimer(device)
        originals = {n: getattr(sqp, n) for n in names}
        for n, fn in originals.items():
            setattr(sqp, n, timer.wrap(fn))
        try:
            g = copy.deepcopy(guess)
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                solution, log = ORIGINAL_SQP_SOLVE(problem, params, g)
            if device == "cuda":
                torch.cuda.synchronize()
            wall = time.perf_counter() - t0
        finally:
            for n, fn in originals.items():
                setattr(sqp, n, fn)
        run = SimpleNamespace(
            solution=solution, log=log, wall=wall, qp=timer.seconds, qp_calls=timer.calls
        )
        if best is None or run.wall < best.wall:
            best = run
    return best


ORIGINAL_SQP_SOLVE = sqp.sqp_solve


def compare(problem, parameters, guess, device, repeats):
    runs = {
        b: solve_with(b, problem, parameters, guess, device, repeats) for b in BACKENDS
    }
    t, b = runs["torch"], runs["bsqp"]
    scale_x = t.solution.x.abs().max().clamp_min(1e-12)
    scale_u = t.solution.u.abs().max().clamp_min(1e-12)
    cost_t = torch.tensor(t.log.total_cost)
    cost_b = torch.tensor(b.log.total_cost)
    result = dict(
        wall={k: r.wall for k, r in runs.items()},
        qp={k: r.qp for k, r in runs.items()},
        sqp_iters={k: r.log.sqp_iterations for k, r in runs.items()},
        admm_iters={k: sum(int(i) for i in r.log.admm_iter_hist) for k, r in runs.items()},
        terminated={k: r.log.envs_terminated for k, r in runs.items()},
        viol={k: max(r.log.constraint_violation) for k, r in runs.items()},
        cost={k: torch.tensor(r.log.total_cost).median().item() for k, r in runs.items()},
        dx_rel=((t.solution.x - b.solution.x).abs().max() / scale_x).item(),
        du_rel=((t.solution.u - b.solution.u).abs().max() / scale_u).item(),
        cost_rel=((cost_t - cost_b).abs() / cost_t.abs().clamp_min(1e-12)).max().item(),
    )
    RESULTS.append(result)
    return t.solution, t.log


RESULTS = []


def load_example(name):
    path = os.path.join(ROOT, "examples", f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"example_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(path))
    spec.loader.exec_module(module)
    return module


def run_example(name, device, batch_size, repeats, check_every, overrides):
    module = load_example(name)

    def patched(problem, parameters, guess):
        parameters.admm_check_every = check_every
        for key, value in overrides.items():
            setattr(parameters, key, value)
        return compare(problem, parameters, guess, device, repeats)

    module.sqp_solve = patched
    args = SimpleNamespace(batch_size=batch_size, device=device, save=None, load=None)
    torch.manual_seed(0)
    RESULTS.clear()
    with contextlib.redirect_stdout(io.StringIO()):
        module.main(args)
    plt.close("all")
    torch.set_default_device("cpu")
    return RESULTS[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--examples", nargs="+", default=EXAMPLES)
    parser.add_argument("--devices", nargs="+", default=["cpu"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--check_every", type=int, default=1)
    parser.add_argument(
        "--set", nargs="*", default=[], metavar="KEY=VALUE",
        help="SqpParameters overrides for both backends, e.g. admm_max_iter=300",
    )
    args = parser.parse_args()
    overrides = {
        k: ast.literal_eval(v) for k, v in (kv.split("=", 1) for kv in args.set)
    }

    print(
        f"threads: torch {torch.get_num_threads()}, "
        f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')}, "
        f"batch {args.batch_size}, bsqp check_every {args.check_every}, "
        f"overrides {overrides}"
    )
    header = (
        f"{'example':<22} {'dev':<4} {'wall t/b [s]':>14} {'QP t/b [s]':>14} "
        f"{'QP x':>6} {'SQP it t/b':>10} {'ADMM it t/b':>12} {'conv t/b':>8} "
        f"{'viol t/b':>17} {'|dx|rel':>8} {'|du|rel':>8} {'cost rel':>8}"
    )
    print(header)
    for device in args.devices:
        for name in args.examples:
            r = run_example(
                name, device, args.batch_size, args.repeats, args.check_every, overrides
            )

            def pair(d, fmt):
                return f"{fmt.format(d['torch'])}/{fmt.format(d['bsqp'])}"

            print(
                f"{name:<22} {device:<4} {pair(r['wall'], '{:.3f}'):>14} "
                f"{pair(r['qp'], '{:.3f}'):>14} {r['qp']['torch'] / r['qp']['bsqp']:>6.1f} "
                f"{pair(r['sqp_iters'], '{}'):>10} {pair(r['admm_iters'], '{}'):>12} "
                f"{pair(r['terminated'], '{}'):>8} {pair(r['viol'], '{:.1e}'):>17} "
                f"{r['dx_rel']:>8.1e} {r['du_rel']:>8.1e} {r['cost_rel']:>8.1e}",
                flush=True,
            )


if __name__ == "__main__":
    main()
