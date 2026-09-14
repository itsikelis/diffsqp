import argparse
import time
import json
import torch
from torch import nn
from mpc import mpc
from mpc.mpc import QuadCost, GradMethods


class QuadrotorDx(nn.Module):
    def __init__(self):
        super().__init__()

        self.n_state = 13  # p(3), q(4), v(3), w(3)
        self.n_ctrl = 4  # f(1), tau(3)

        # Aligned parameters: mass, gravity
        self.params = torch.tensor([0.1, 9.81])

        # Exact Inertia Matrix from turbompc/diffsqp
        self.inertia = torch.tensor(
            [
                [0.1, 0.01, 0.01],
                [0.01, 0.1, 0.01],
                [0.01, 0.01, 0.1],
            ]
        )

        self.dt = 0.01

        # Actuation limits (Unbounded to match turbompc/diffsqp)
        self.lower = torch.tensor([-2.0, -10.0, -10.0, -10.0])
        self.upper = torch.tensor([2.0, 10.0, 10.0, 10.0])

    @staticmethod
    def quat_to_rot(q: torch.Tensor) -> torch.Tensor:
        qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        qw2, qx2, qy2, qz2 = qw * qw, qx * qx, qy * qy, qz * qz

        row0 = torch.stack(
            [qw2 + qx2 - qy2 - qz2, 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            dim=-1,
        )
        row1 = torch.stack(
            [2 * (qx * qy + qw * qz), qw2 - qx2 + qy2 - qz2, 2 * (qy * qz - qw * qx)],
            dim=-1,
        )
        row2 = torch.stack(
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), qw2 - qx2 - qy2 + qz2],
            dim=-1,
        )
        return torch.stack([row0, row1, row2], dim=-2)

    @staticmethod
    def q_left(q: torch.Tensor) -> torch.Tensor:
        qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        row0 = torch.stack([qw, -qx, -qy, -qz], dim=-1)
        row1 = torch.stack([qx, qw, -qz, qy], dim=-1)
        row2 = torch.stack([qy, qz, qw, -qx], dim=-1)
        row3 = torch.stack([qz, -qy, qx, qw], dim=-1)
        return torch.stack([row0, row1, row2, row3], dim=-2)

    @staticmethod
    def S(u: torch.Tensor) -> torch.Tensor:
        z = torch.zeros_like(u[..., 0])
        row0 = torch.stack([z, -u[..., 2], u[..., 1]], dim=-1)
        row1 = torch.stack([u[..., 2], z, -u[..., 0]], dim=-1)
        row2 = torch.stack([-u[..., 1], u[..., 0], z], dim=-1)
        return torch.stack([row0, row1, row2], dim=-2)

    def forward(self, state, u):
        squeeze = state.ndimension() == 1
        if squeeze:
            state = state.unsqueeze(0)
            u = u.unsqueeze(0)

        m = self.params[0]
        g = self.params[1]

        inertia = self.inertia.to(device=state.device, dtype=state.dtype)
        inertia_inv = torch.inverse(inertia)

        # Clamp inputs
        f = torch.clamp(u[..., 0:1], self.lower[0].item(), self.upper[0].item())
        tau = torch.clamp(u[..., 1:4], self.lower[1].item(), self.upper[1].item())

        q = state[..., 3:7]
        v = state[..., 7:10]
        w = state[..., 10:13]

        q = q / torch.norm(q, dim=-1, keepdim=True)
        e3 = torch.tensor([0.0, 0.0, 1.0], device=state.device, dtype=state.dtype)
        R = self.quat_to_rot(q)

        def bmv(mat, vec):
            return (mat @ vec.unsqueeze(-1)).squeeze(-1)

        # Continuous dynamics
        p_dot = v
        v_dot = -g * e3 + (f / m) * bmv(R, e3)
        zero_w = torch.cat([torch.zeros_like(w[..., 0:1]), w], dim=-1)
        q_dot = 0.5 * bmv(self.q_left(q), zero_w)
        Jw = bmv(inertia, w)
        w_dot = bmv(inertia_inv, bmv(self.S(Jw), w) + tau)

        # Euler integration step
        next_p = state[..., 0:3] + self.dt * p_dot
        next_q = q + self.dt * q_dot
        next_v = v + self.dt * v_dot
        next_w = w + self.dt * w_dot

        next_q = next_q / torch.norm(next_q, dim=-1, keepdim=True)
        next_state = torch.cat([next_p, next_q, next_v, next_w], dim=-1)

        if squeeze:
            next_state = next_state.squeeze(0)
        return next_state


def main(args):
    torch.manual_seed(time.time())
    device = args.device
    n_batch = args.batch_size

    torch.set_default_device(device)

    dx = QuadrotorDx().to(device)
    mpc_T = 100  # Full trajectory optimization horizon

    # 1. Initial State (Hovering at origin)
    xinit = torch.tensor(
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=device
    ).repeat(n_batch, 1)

    # 1.5 Randomise Initial state (Matching other solvers)
    noise_std = [0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.01, 0.01, 0.01, 0.01, 0.0, 0.0, 0.0]
    noise_std_tensor = torch.tensor(noise_std, device=device)

    xinit += noise_std_tensor * torch.randn((n_batch, dx.n_state), device=device)
    quats = xinit[:, 3:7]
    xinit[:, 3:7] = quats / torch.norm(quats, p=2, dim=1, keepdim=True)

    # 2. Target State & Control
    x_des = torch.tensor(
        [1.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=device
    )
    u_des = torch.tensor([0.981, 0.0, 0.0, 0.0], device=device)
    target = torch.cat((x_des, u_des))

    # 3. Cost Weights mapped to [Pos(3), Quat(4), Vel(3), Omega(3)] + [Thrust(1), Moments(3)]
    q_w = torch.tensor([1e-4] * 3 + [1e-4] * 4 + [1e-1] * 3 + [1e-1] * 3, device=device)
    r_w = torch.tensor([1e1, 1e-1, 1e-1, 1e-1], device=device)
    qf_w = torch.tensor([1e5] * 3 + [1e3] * 4 + [1e3] * 3 + [1e3] * 3, device=device)

    # Construct Stage Cost Q and p (Multiply by 2 for mpc.pytorch 1/2 formulation)
    Q_stage = torch.diag(torch.cat((q_w, r_w)))
    p_stage = -Q_stage @ target

    # Construct Terminal Cost Qf and pf
    Q_term = torch.diag(torch.cat((qf_w, r_w)))
    p_term = -Q_term @ target

    # Replicate over horizon (T, B, dim, dim)
    Q_seq = Q_stage.unsqueeze(0).unsqueeze(0).repeat(mpc_T, n_batch, 1, 1)
    p_seq = p_stage.unsqueeze(0).unsqueeze(0).repeat(mpc_T, n_batch, 1)

    # Override the final step with terminal costs
    Q_seq[-1] = Q_term.unsqueeze(0).repeat(n_batch, 1, 1)
    p_seq[-1] = p_term.unsqueeze(0).repeat(n_batch, 1)

    # 4. Control Bounds expanding over horizon
    u_lower_expanded = (
        dx.lower.unsqueeze(0).unsqueeze(0).repeat(mpc_T, n_batch, 1).to(device)
    )
    u_upper_expanded = (
        dx.upper.unsqueeze(0).unsqueeze(0).repeat(mpc_T, n_batch, 1).to(device)
    )

    # 5. Initial Guess (Hover thrust)
    u_init = u_des.unsqueeze(0).unsqueeze(0).repeat(mpc_T, n_batch, 1)

    # 6. Solve (Single Trajectory Optimization)
    print("Solving Quadrotor Trajopt Task with mpc_pytorch...")
    start_time = time.time()

    x, u, costs, iters = mpc.MPC(
        dx.n_state,
        dx.n_ctrl,
        mpc_T,
        u_init=u_init,
        u_lower=u_lower_expanded,
        u_upper=u_upper_expanded,
        lqr_iter=50,  # Matched max_iter to other solvers
        verbose=0,
        eps=1e-4,  # Matches tolerance
        n_batch=n_batch,
        linesearch_decay=0.5,
        max_linesearch_iter=10,
        exit_unconverged=False,
        detach_unconverged=False,
        grad_method=GradMethods.AUTO_DIFF,
        backprop=False,
    )(xinit, QuadCost(Q_seq, p_seq), dx)

    solve_time = time.time() - start_time

    # --- 7. Constraint Violation Calculation ---
    with torch.no_grad():
        # Evaluate states sequentially to find dynamic feasibility
        # mpc.pytorch yields x and u of length mpc_T.
        # x[t+1] corresponds to applying u[t] to x[t].
        T_sim = x.shape[0] - 1
        x_curr = x[:T_sim]
        u_curr = u[:T_sim]

        # Simulate forward with returned controls
        x_next_pred = dx(x_curr, u_curr)
        x_next_actual = x[1 : T_sim + 1]

        # Dynamics violation (Maximum absolute deviation between solver state and physics)
        dyn_violation = torch.max(torch.abs(x_next_actual - x_next_pred)).item()

        # Control Bound Violation
        lower_bound = dx.lower.view(1, 1, -1).to(device)
        upper_bound = dx.upper.view(1, 1, -1).to(device)

        lower_violation = torch.max(torch.relu(lower_bound - u)).item()
        upper_violation = torch.max(torch.relu(u - upper_bound)).item()
        ctrl_violation = max(lower_violation, upper_violation)

        max_violation = max(dyn_violation, ctrl_violation)

    print(f"Solve Time: {solve_time:.4f}s")
    print(f"Max Dynamics Violation: {dyn_violation:.2e}")
    print(f"Max Control Violation: {ctrl_violation:.2e}")
    print(f"Total Max Constraint Violation: {max_violation:.2e}")
    # ---------------------------------------------

    # 8. Safe Logging
    log = {
        "final_state": x[-1, :, :].tolist(),
        "sqp_iterations": iters,
        "solve_time": solve_time,
        "final_objective": costs.tolist(),
        "constraint_violation": max_violation,
        "dynamics_violation": dyn_violation,
        "control_violation": ctrl_violation,
    }

    if args.save:
        print(f"Saving log to {args.save}.json...")
        with open(args.save + ".json", "w", encoding="utf-8") as f:
            json.dump(log, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-batch_size", type=int, help="Batch size", default=1)
    parser.add_argument("-device", type=str, help="Device to run on", default="cuda")
    parser.add_argument("-save", type=str, help="Filename to save result")
    parser.add_argument("-load", type=str, help="Filename to load result")
    main(parser.parse_args())
