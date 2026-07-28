from abc import ABC, abstractmethod
from typing import List, Optional
import torch
from torch import relu

from diffsqp.costs import Cost
from diffsqp.dynamics import Dynamics
from diffsqp.constraints import UnderactuationConstraint, GenericConstraint
from diffsqp.types import SqpSolution, QpParameters

from diffsqp.utils.math import mv


class ProblemParameters:
    def __init__(self, **args):
        # self.system: str = args["name"]
        self.inverse_dynamics: bool = args["inverse_dynamics"]
        self.n_batch: int = args["n_batch"]
        self.tf: float = args["tf"]
        self.dt: float = args["dt"]
        self.n_x: int = len(args["q_w"])
        self.n_u: int = len(args["r_w"])  # Number of underactuated DoFs
        self.n_h: int = args["n_h"]
        self.horizon = int(self.tf / self.dt)
        # # Initial and final states
        self.x_init = torch.tensor(args["x_init"]).repeat(self.n_batch, 1)
        # Apply noise only to the first two dimensions (usually positions)
        self.x_init[:, 0:2] += args["noise_std"] * torch.randn((self.n_batch, 2))
        self.x_des = torch.tensor(args["x_des"]).repeat(self.n_batch, 1)

        # State-control bounds
        self.x_lb = torch.tensor(args["x_lb"])
        self.x_ub = torch.tensor(args["x_ub"])
        self.u_lb = torch.tensor(args["u_lb"])
        self.u_ub = torch.tensor(args["u_ub"])

        # # Cost weights
        self.q_w = torch.tensor(args["q_w"])
        self.r_w = torch.tensor(args["r_w"])
        self.qf_w = torch.tensor(args["qf_w"])


class Problem(ABC):
    """
    An abstract base class representing a Trajectory Optimization problem.
    """

    def __init__(self, params: ProblemParameters) -> None:
        self.inverse_dynamics = params.inverse_dynamics
        self.horizon = params.horizon
        self.dt = params.dt
        self.n_x = params.n_x
        self.n_u = params.n_u
        self.n_h = params.n_h
        self.n_batch = params.n_batch

        self.costs: List[List[Cost]] = []
        self.dynamics: Dynamics = None
        self.underactuation: UnderactuationConstraint = None
        self.constraints: List[List[GenericConstraint]] = [None] * self.horizon

        # Initialize gradient tensors
        self.Lx = torch.zeros((self.n_batch, self.horizon, self.n_x))
        self.Lu = torch.zeros((self.n_batch, self.horizon - 1, self.n_u))

    # --- Cost Aggregation Methods ---

    def l(
        self, stage_idx: int, x: torch.Tensor, u: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute the total stage cost by summing all cost components.

        Args:
            stage_idx: The current time step index.
            x: State tensor [nB x n_x].
            u: Control tensor [nB x n_u]. Optional for terminal stage.

        Returns:
            Total scalar cost per batch [nB].
        """
        all_costs = torch.stack([c.l(x, u) for c in self.costs[stage_idx]])
        return torch.sum(all_costs, dim=0)

    def lx(
        self, stage_idx: int, x: torch.Tensor, u: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Total state gradient of the cost at stage_idx."""
        all_grads = torch.stack([c.lx(x, u) for c in self.costs[stage_idx]])
        return torch.sum(all_grads, dim=0)

    def lu(self, stage_idx: int, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Total control gradient of the cost at stage_idx."""
        all_grads = torch.stack([c.lu(x, u) for c in self.costs[stage_idx]])
        return torch.sum(all_grads, dim=0)

    def lxx(
        self, stage_idx: int, x: torch.Tensor, u: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Total state Hessian (d^2L/dx^2) at stage_idx."""
        all_hessians = torch.stack([c.lxx(x, u) for c in self.costs[stage_idx]])
        return torch.sum(all_hessians, dim=0)

    def luu(self, stage_idx: int, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Total control Hessian (d^2L/du^2) at stage_idx."""
        all_hessians = torch.stack([c.luu(x, u) for c in self.costs[stage_idx]])
        return torch.sum(all_hessians, dim=0)

    def lux(self, stage_idx: int, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Total cross-derivative (d^2L/dudx) at stage_idx."""
        all_hessians = torch.stack([c.lux(x, u) for c in self.costs[stage_idx]])
        return torch.sum(all_hessians, dim=0)

    def lxu(self, stage_idx: int, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Total cross-derivative (d^2L/dxdu) at stage_idx."""
        all_hessians = torch.stack([c.lxu(x, u) for c in self.costs[stage_idx]])
        return torch.sum(all_hessians, dim=0)

    # --- Constraint Aggregation Methods ---

    def n_g(self, stage_idx):
        if self.constraints[stage_idx] is None:
            return 0

        return sum((c.n_g for c in self.constraints[stage_idx]))

    def g(
        self, stage_idx: int, x: torch.Tensor, u: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Concatenate all stage constraints into a single vector.

        Returns:
            A tensor of concatenated constraints [nB x total_constraints].
        """
        if self.constraints[stage_idx] is None:
            return None

        if u is not None:
            constr = torch.cat([c.g(x, u) for c in self.constraints[stage_idx]], dim=1)
        else:
            print(stage_idx)
            constr = torch.cat([c.g(x) for c in self.constraints[stage_idx]], dim=1)

        return constr

    def g_bounds(self, stage_idx: int) -> torch.Tensor:
        if self.constraints[stage_idx] is None:
            return None, None

        lb = torch.cat([c.lb for c in self.constraints[stage_idx]])
        ub = torch.cat([c.ub for c in self.constraints[stage_idx]])
        return lb, ub

    def gx(
        self, stage_idx: int, x: torch.Tensor, u: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Jacobian of the aggregated constraints with respect to state x."""
        grad = torch.cat([c.gx(x, u) for c in self.constraints[stage_idx]], dim=1)
        return grad

    def gu(self, stage_idx: int, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Jacobian of the aggregated constraints with respect to control u."""
        grad = torch.cat([c.gu(x, u) for c in self.constraints[stage_idx]], dim=1)
        return grad

    def evaluate_guess(self, solution_guess: SqpSolution):
        """Return total trajectory cost and constraint violation"""
        batch_size = self.n_batch
        horizon = self.horizon
        dt = self.dt

        cost = torch.zeros((batch_size))
        convergence_error = torch.zeros((batch_size))

        # Calculate total trajectory cost
        max_generic_violation = torch.zeros((batch_size))
        for k in range(horizon - 1):
            cost += self.l(k, solution_guess.x[:, k], solution_guess.u[:, k])
            g_val = self.g(k, solution_guess.x[:, k], solution_guess.u[:, k])
            lb, ub = self.g_bounds(k)
            if g_val is not None:
                stage_error = torch.cat([relu(lb - g_val), relu(g_val - ub)], dim=1)
                convergence_error = torch.maximum(
                    convergence_error, stage_error.max(dim=1).values
                )
        # Final stage
        cost += self.l(-1, solution_guess.x[:, -1])
        g_val = self.g(-1, solution_guess.x[:, -1])
        lb, ub = self.g_bounds(-1)
        if g_val is not None:
            stage_error = torch.cat([relu(g_val - ub), relu(lb - g_val)], dim=1)
            convergence_error = torch.maximum(
                convergence_error, stage_error.max(dim=1).values
            )

        # Dynamics violation
        x_next = solution_guess.x[:, 1:]
        x_curr = solution_guess.x[:, :-1]
        u_curr = solution_guess.u[:]
        dynamics_violations = x_next - self.dynamics.f(x_curr, u_curr, dt)
        max_dynamics_violation = torch.norm(
            dynamics_violations, p=float("inf"), dim=[1, 2]
        )
        convergence_error = torch.maximum(convergence_error, max_dynamics_violation)

        # Underactuation violation
        if self.underactuation is not None:
            uact_violation = self.underactuation.h(x_curr, u_curr)
            max_uact_violation = torch.norm(uact_violation, p=float("inf"), dim=[1, 2])
            convergence_error = torch.maximum(convergence_error, max_uact_violation)

        return cost, convergence_error

    def linearize(self, solution_guess: SqpSolution, regularization_scale):
        batch_size = self.n_batch
        horizon = self.horizon
        n_x, n_u = self.n_x, self.n_u
        n_h = self.n_h

        Q = torch.zeros((batch_size, horizon, n_x, n_x))
        q = torch.zeros((batch_size, horizon, n_x))
        R = torch.zeros((batch_size, horizon - 1, n_u, n_u))
        r = torch.zeros((batch_size, horizon - 1, n_u))
        S = torch.zeros((batch_size, horizon - 1, n_u, n_x))

        A = torch.zeros((batch_size, horizon - 1, n_x, n_x))
        B = torch.zeros((batch_size, horizon - 1, n_x, n_u))
        b = torch.zeros((batch_size, horizon - 1, n_x))

        C = None
        D = None
        d = None
        if self.underactuation is not None:
            n_h = self.n_h
            C = torch.zeros((batch_size, horizon - 1, n_h, n_x))
            D = torch.zeros((batch_size, horizon - 1, n_h, n_u))
            d = torch.zeros((batch_size, horizon - 1, n_h))

        M = None
        N = None
        n = None
        if self.constraints is not None:
            M = [None] * horizon
            N = [None] * (horizon - 1)
            n = [None] * horizon

        # Fill matrices
        for k in range(horizon - 1):
            x_lin, u_lin, x_next = (
                solution_guess.x[:, k],
                solution_guess.u[:, k],
                solution_guess.x[:, k + 1],
            )

            A[:, k] = self.dynamics.fx(x_lin, u_lin, self.dt)
            B[:, k] = self.dynamics.fu(x_lin, u_lin, self.dt)
            b[:, k] = self.dynamics.f(x_lin, u_lin, self.dt) - x_next

            Q[:, k] = self.lxx(k, x_lin, u_lin) + regularization_scale * torch.eye(n_x)
            q[:, k] = self.lx(k, x_lin, u_lin)
            R[:, k] = self.luu(k, x_lin, u_lin) + regularization_scale * torch.eye(n_u)
            r[:, k] = self.lu(k, x_lin, u_lin)
            S[:, k] = self.lux(k, x_lin, u_lin)

            # Underactuation augmentation
            if self.underactuation is not None:
                C[:, k] = self.underactuation.hx(x_lin, u_lin)
                D[:, k] = self.underactuation.hu(x_lin, u_lin)
                d[:, k] = self.underactuation.h(x_lin, u_lin)

            if self.constraints[k] is not None:
                M[k] = self.gx(k, x_lin, u_lin)
                N[k] = self.gu(k, x_lin, u_lin)
                n[k] = self.g(k, x_lin, u_lin)

        x_F = solution_guess.x[:, -1]
        Q[:, -1] = self.lxx(-1, x_F)
        q[:, -1] = self.lx(-1, x_F)

        if self.constraints[-1] is not None:
            M[-1] = self.gx(-1, x_F)
            n[-1] = self.g(-1, x_F)

        return QpParameters(
            Q=Q, q=q, R=R, r=r, S=S, A=A, B=B, b=b, C=C, D=D, d=d, M=M, N=N, n=n
        )
