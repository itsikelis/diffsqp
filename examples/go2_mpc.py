"""MPC-style simulation of the Go2 standing problem.

The problem of `examples/go2_inverse_riccati.py` (or `go2_inverse_admm.py`
with `--formulation admm`), over the first `--horizon` nodes of the 1 s base
reference, is solved once to convergence. Then, every `--dt_mpc` seconds, the
solution, shifted in time and with its first state set to the measured one,
warm starts a single SQP iteration (no line search, a few ADMM iterations from
the shifted inequality multipliers) over a window that moves along the
reference and holds its last node at the end. The joint torques of its first
control, tau[6:] = (RNEA(q, v, a) - sum_c J_c^T f_c)[6:], drive a simulated robot.

Each problem of the batch is a separate robot. The simulation uses the forward
dynamics of gdyn (`gdyn.aba`) with the feet in sticking contact: the contact
forces are the ones that bring the foot velocities to zero at the end of the
step (a velocity-level, semi-implicit Euler contact model with a small
position correction). They are not limited to the friction cones; the run
reports how far they are from leaving them.

    OMP_NUM_THREADS=1 uv run examples/go2_mpc.py --batch_size 4 --save out/go2
    OMP_NUM_THREADS=1 uv run examples/go2_mpc.py --batch_size 1 --horizon 10 \\
        --admm_iters 100 --fixed_admm_iters
    uv run visuals/visualize_trajectory.py --urdf go2.urdf --floating_base --quat_order xyzw \\
        --show_target --floor_z -0.022 --dt 0.01 --file out/go2_mpc.pt --batch_size 4
"""

import argparse
import importlib
import time

import torch
import gdyn

from diffsqp.solvers import sqp_solve


def contact_forward_dynamics(model, ex, q, v, tau, p0, dt, correction=0.2):
    """
    Accelerations (B, nv) and world-frame foot forces (B, 4, 3) of the robot
    with sticking feet, for the generalized forces tau (zero base rows).

    aba is affine in tau, so a = a_free + M^-1 L^T f with a_free = aba(q, v, tau)
    and the columns of M^-1 L^T from 12 more aba calls. The forces solve
    L (v + dt a) = -correction (p - p0) / dt.
    """
    B, nq, nv = q.shape[0], q.shape[-1], v.shape[-1]
    p, L, _ = ex.foot_kinematics(model, q)
    L = L.flatten(-3, -2)  # (B, 12, nv)
    a_free = gdyn.aba(model, q, v, tau)
    taus = (tau.unsqueeze(-2) + L).reshape(-1, nv)
    a_unit = gdyn.aba(
        model,
        q.unsqueeze(-2).expand(B, 12, nq).reshape(-1, nq),
        v.unsqueeze(-2).expand(B, 12, nv).reshape(-1, nv),
        taus,
    ).reshape(B, 12, nv)
    Minv_LT = (a_unit - a_free.unsqueeze(-2)).mT  # (B, nv, 12)
    target = -correction * (p - p0).flatten(-2) / dt
    rhs = (target - (L @ v.unsqueeze(-1)).squeeze(-1)) / dt - (L @ a_free.unsqueeze(-1)).squeeze(-1)
    f = torch.linalg.solve(L @ Minv_LT, rhs)
    a = a_free + (Minv_LT @ f.unsqueeze(-1)).squeeze(-1)
    return a, f.unflatten(-1, (4, 3))


def shift(tensor: torch.Tensor, shifts: int) -> torch.Tensor:
    """Advance a (B, H, ...) horizon by `shifts` stages, repeating the last."""
    return torch.cat([tensor[:, shifts:], tensor[:, -1:].expand_as(tensor[:, :shifts])], dim=1)


def shift_stages(tensor: torch.Tensor, shifts: int) -> torch.Tensor:
    """`shift` for per-node data whose rows differ at the terminal node: the
    stages shift among themselves and the terminal node keeps its own."""
    return torch.cat([shift(tensor[:, :-1], shifts), tensor[:, -1:]], dim=1)


def move_horizon(setup, first_node: int):
    """Point the costs at the reference nodes first_node, first_node + 1, ...,
    holding the last node of the reference past its end."""
    last_node = setup.x_ref.shape[1] - 1
    costs = [*setup.stage_costs, setup.terminal_cost]
    for k, cost in enumerate(costs):
        cost.x_des = setup.x_ref[:, min(first_node + k, last_node)]


def synchronize(device: str):
    if device != "cpu":
        torch.cuda.synchronize()


def main(args):
    if args.device != "cpu":
        torch.set_default_device(args.device)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    ex = importlib.import_module(f"go2_inverse_{args.formulation}")
    batch_size = args.batch_size

    model = gdyn.load_urdf(ex.URDF_PATH)
    setup = ex.build_problem(model, batch_size, args.horizon)
    problem, params = setup.problem, setup.sqp_parameters
    if args.admm_iters_initial is not None:
        params.admm_max_iter = args.admm_iters_initial
    x_ref, p0, tau_max = setup.x_ref, setup.p0.repeat(batch_size, 1, 1), setup.tau_max
    x_des = x_ref[:, -1]
    dt_solver, horizon = problem.dt, problem.horizon
    last_node = x_ref.shape[1] - 1

    print(
        f"Solving the initial trajectories ({batch_size} problems, {args.formulation}, "
        f"{horizon} of {last_node + 1} nodes)..."
    )
    synchronize(args.device)
    t0 = time.perf_counter()
    solution, log = sqp_solve(problem, params, setup.initial_guess)
    synchronize(args.device)
    print(
        f"Initial solve: {log.sqp_iterations} SQP iterations, "
        f"{log.envs_terminated}/{batch_size} converged, {time.perf_counter() - t0:.3f} s"
    )

    # MPC: one SQP iteration, full step, a few ADMM iterations
    params.sqp_max_iter = 1
    params.ls_max_iter = 0
    params.admm_max_iter = args.admm_iters
    params.verbose = False
    params.admm_warm_start_duals = not args.cold_duals
    if args.fixed_admm_iters:
        params.admm_abs_tolerance = 0.0
        params.admm_rel_tolerance = 0.0

    x = solution.x[:, 0].clone()  # true states
    states, torques, forces = [x.cpu()], [], []
    solve_times = []
    tracking_errors = []
    elapsed_shifts, accumulated_time = 0, 0.0
    dt_sim = args.dt_mpc / args.substeps

    print(f"MPC loop: {args.n_mpc} steps of {args.dt_mpc} s ({args.substeps} simulation substeps)")
    for step in range(args.n_mpc):
        # Warm-started single SQP iteration from the measured state
        solution.x[:, 0] = x
        synchronize(args.device)
        t0 = time.perf_counter()
        solution, log = sqp_solve(problem, params, solution)
        synchronize(args.device)
        solve_times.append(time.perf_counter() - t0)

        # Motor torques of the first control, held over the MPC step
        tau = ex.inverse_dynamics(model, x, solution.u[:, 0])
        tau = torch.cat([torch.zeros_like(tau[:, :6]), torch.clamp(tau[:, 6:], -tau_max, tau_max)], dim=-1)
        torques.append(tau[:, 6:].cpu())
        for _ in range(args.substeps):
            q = ex.normalized(x[:, :19])
            a, f = contact_forward_dynamics(model, ex, q, x[:, 19:], tau, p0, dt_sim)
            x = problem.dynamics.f(x, torch.cat([a, f.flatten(-2)], dim=-1), dt_sim)
        states.append(x.cpu())
        forces.append(f.cpu())

        # Shift the warm start and the references by the elapsed solver steps
        accumulated_time += args.dt_mpc
        shifts = int((accumulated_time + 1e-6) // dt_solver)
        if 0 < shifts < horizon:
            accumulated_time -= shifts * dt_solver
            elapsed_shifts += shifts
            solution.x = shift(solution.x, shifts)
            solution.u = shift(solution.u, shifts)
            solution.mu = shift(solution.mu, shifts)
            solution.nu = shift(solution.nu, shifts)
            if torch.is_tensor(solution.ksi):
                solution.ksi = shift_stages(solution.ksi, shifts)
            move_horizon(setup, elapsed_shifts)
            # On a reference node, the errors are free of the time quantization
            tracking_errors.append(pose_errors(ex, x, x_ref[:, min(elapsed_shifts, last_node)]))

        if args.verbose:
            base_error = torch.norm(x[:, :3] - x_ref[:, min(elapsed_shifts, last_node), :3], dim=-1)
            print(
                f"MPC step {step + 1}/{args.n_mpc} | solve {1e3 * solve_times[-1]:.1f} ms | "
                f"base tracking error {base_error.max().item():.2e} m"
            )

    x_now = x_ref[:, min(elapsed_shifts, last_node)]
    report(
        ex, model, x, x_now, p0, tau_max, torch.stack(torques, 1), torch.stack(forces, 1),
        solve_times, tracking_errors, args.dt_mpc,
    )

    if args.save:
        path = f"{args.save}_mpc.pt"
        torch.save(
            {
                "x": torch.stack(states, dim=1),
                "u": torch.stack(torques, dim=1),
                "f": torch.stack(forces, dim=1),
                "x_des": x_des.cpu(),
                "dt_mpc": args.dt_mpc,
            },
            path,
        )
        print(f"MPC trajectories saved to {path}")


def pose_errors(ex, x, x_now):
    """Base position (m) and rotation (rad) errors, (B, 2), against the reference state x_now."""
    q = ex.normalized(x[:, :19])
    pos_error = torch.norm(q[:, :3] - x_now[:, :3], dim=-1)
    quat_dot = torch.sum(q[:, 3:7] * x_now[:, 3:7], dim=-1).abs().clamp(max=1.0)
    return torch.stack([pos_error, 2.0 * torch.acos(quat_dot)], dim=-1).cpu()


def report(ex, model, x, x_now, p0, tau_max, torques, forces, solve_times, tracking_errors, dt_mpc):
    times = 1e3 * torch.tensor(solve_times)
    steady = times[1:] if len(times) > 1 else times
    print(
        f"MPC solve time: mean {steady.mean():.2f} ms, median {steady.median():.2f} ms, "
        f"p99 {steady.quantile(0.99):.2f} ms, max {steady.max():.2f} ms "
        f"(first step {times[0]:.2f} ms, excluded)"
    )
    missed = (steady > 1e3 * dt_mpc).sum().item()
    print(f"Solves over the {1e3 * dt_mpc:.0f} ms period: {missed}/{len(steady)}")

    # Against the reference at the current time
    final_error = pose_errors(ex, x, x_now).amax(dim=0)
    print(f"Base position tracking error (max over batch): {final_error[0].item():.2e} m")
    print(f"Base rotation tracking error (max over batch): {final_error[1].item():.2e} rad")
    if tracking_errors:
        errors = torch.stack(tracking_errors)
        peak, rms = errors.amax(dim=(0, 1)), errors.square().mean(dim=(0, 1)).sqrt()
        print(f"Base position error over the run: RMS {rms[0].item():.2e} m, peak {peak[0].item():.2e} m")
        print(f"Base rotation error over the run: RMS {rms[1].item():.2e} rad, peak {peak[1].item():.2e} rad")
    q = ex.normalized(x[:, :19])
    p, _, _ = ex.foot_kinematics(model, q)
    print(f"Final foot drift          (max over batch): {torch.norm(p - p0, dim=-1).max().item():.2e} m")
    print(f"Peak |tau| / tau_max      (max over batch): {(torques.abs() / tau_max.cpu()).max().item():.2f}")
    fz = forces[..., 2]
    tangential = torch.norm(forces[..., :2], dim=-1)
    print(f"Min normal foot force     (min over batch): {fz.min().item():.2f} N")
    print(f"Max |f_t| / f_z           (max over batch): {(tangential / fz.clamp(min=1e-6)).max().item():.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4, help="Number of robots")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--formulation", choices=["riccati", "admm"], default="riccati")
    parser.add_argument("--horizon", type=int, default=50, help="Nodes of the moving window (0.02 s apart)")
    parser.add_argument("--seed", type=int, help="Random seed of the pose offsets")
    parser.add_argument("--dt_mpc", type=float, default=0.01, help="MPC period (s)")
    parser.add_argument("--n_mpc", type=int, default=150, help="Number of MPC steps")
    parser.add_argument("--substeps", type=int, default=4, help="Simulation steps per MPC step")
    parser.add_argument("--admm_iters", type=int, default=10, help="ADMM iterations per MPC solve")
    parser.add_argument(
        "--fixed_admm_iters", action="store_true", help="Always run --admm_iters iterations (no early termination)"
    )
    parser.add_argument("--cold_duals", action="store_true", help="Start each MPC solve from zero multipliers")
    parser.add_argument(
        "--admm_iters_initial", type=int, help="ADMM iterations per QP of the initial solve (default: the example's)"
    )
    parser.add_argument("--verbose", action="store_true", help="Print every MPC step")
    parser.add_argument("--save", type=str, help="Save path prefix (writes <prefix>_mpc.pt)")
    main(parser.parse_args())
