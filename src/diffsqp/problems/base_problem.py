from abc import ABC, abstractmethod
from typing import List, Optional
import torch
from torch import relu

from diffsqp.costs import Cost
from diffsqp.dynamics import Dynamics
from diffsqp.constraints import UnderactuationConstraint, GenericConstraint
from diffsqp.types import SqpSolution, AdmmSolution, QpParameters

from diffsqp.utils.math import mv

# Bound used for padded constraint rows (stages with fewer constraints than
# the largest stage). Finite, so 0 * bound stays 0 in complementarity checks.
PAD_BOUND = 1e20


class ProblemParameters:
    def __init__(self, **args):
        # self.system: str = args["name"]
        self.inverse_dynamics: bool = args["inverse_dynamics"]
        self.batch_size: int = args["batch_size"]
        self.tf: float = args["tf"]
        self.dt: float = args["dt"]
        self.n_h: int = args["n_h"]
        self.horizon = int(self.tf / self.dt)
        # # Initial and final states
        self.x_init = torch.tensor(args["x_init"])
        self.x_des = torch.tensor(args["x_des"])
        self.noise_std = args["noise_std"]

        # State-control bounds
        self.x_lb = torch.tensor(args["x_lb"])
        self.x_ub = torch.tensor(args["x_ub"])
        self.u_lb = torch.tensor(args["u_lb"])
        self.u_ub = torch.tensor(args["u_ub"])

        # # Cost weights
        self.q_w = torch.tensor(args["q_w"])
        self.r_w = torch.tensor(args["r_w"])
        self.qf_w = torch.tensor(args["qf_w"])

    def __str__(self) -> str:
        return (
            f"=== Problem Parameters ===\n"
            f"  Inv Dynamics            : {self.inverse_dynamics}\n"
            f"  Batch Size              : {self.batch_size}\n"
            f"  Final Time (tf)         : {self.tf:.3f}\n"
            f"  Time Step (dt)          : {self.dt:.4f}\n"
            f"  Horizon Steps           : {self.horizon}\n"
            f"  Underactuation Dim (nh) : {self.n_h}\n"
            f"--------------------------\n"
            f"  State bounds (lower)    : {self.x_lb}\n"
            f"  State bounds (upper)    : {self.x_ub}\n"
            f"  Control bounds (lower)  : {self.u_lb}\n"
            f"  Control bounds (lower)  : {self.u_ub}\n"
            f"--------------------------\n"
            f"  Initial state           : {self.x_init}\n"
            f"  Desired state           : {self.x_des}\n"
            f"=========================="
        )


class Problem(ABC):
    """
    An abstract base class representing a Trajectory Optimization problem.
    """

    def __init__(self, params: ProblemParameters, sys_params) -> None:
        self.inverse_dynamics = params.inverse_dynamics
        self.horizon = params.horizon
        self.dt = params.dt
        self.n_x = sys_params.n_x
        self.n_u = sys_params.n_u
        self.n_h = params.n_h
        self.batch_size = params.batch_size

        self.costs: List[List[Cost]] = []
        self.dynamics: Dynamics = None
        self.underactuation: UnderactuationConstraint = None
        self.constraints: List[List[GenericConstraint]] = [None] * self.horizon

        # Initialize gradient tensors
        self.Lx = torch.zeros((self.batch_size, self.horizon, self.n_x))
        self.Lu = torch.zeros((self.batch_size, self.horizon - 1, self.n_u))

        # Padded stage bounds, see `padded_bounds`
        self._padded_bounds = None

    # --- Cost Aggregation Methods ---

    def _sum_costs(self, method: str, stage_idx: int, x, u):
        costs = self.costs[stage_idx]
        if len(costs) == 1:
            return getattr(costs[0], method)(x, u)
        return torch.sum(torch.stack([getattr(c, method)(x, u) for c in costs]), dim=0)

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
        return self._sum_costs("l", stage_idx, x, u)

    def lx(
        self, stage_idx: int, x: torch.Tensor, u: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Total state gradient of the cost at stage_idx."""
        return self._sum_costs("lx", stage_idx, x, u)

    def lu(self, stage_idx: int, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Total control gradient of the cost at stage_idx."""
        return self._sum_costs("lu", stage_idx, x, u)

    def lxx(
        self, stage_idx: int, x: torch.Tensor, u: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Total state Hessian (d^2L/dx^2) at stage_idx."""
        return self._sum_costs("lxx", stage_idx, x, u)

    def luu(self, stage_idx: int, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Total control Hessian (d^2L/du^2) at stage_idx."""
        return self._sum_costs("luu", stage_idx, x, u)

    def lux(self, stage_idx: int, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Total cross-derivative (d^2L/dudx) at stage_idx."""
        return self._sum_costs("lux", stage_idx, x, u)

    def lxu(self, stage_idx: int, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Total cross-derivative (d^2L/dxdu) at stage_idx."""
        return self._sum_costs("lxu", stage_idx, x, u)

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
            constr = torch.cat([c.g(x) for c in self.constraints[stage_idx]], dim=1)

        return constr

    def g_bounds(self, stage_idx: int) -> torch.Tensor:
        if self.constraints[stage_idx] is None:
            return None, None

        lb = torch.cat([c.lb for c in self.constraints[stage_idx]])
        ub = torch.cat([c.ub for c in self.constraints[stage_idx]])
        return lb, ub

    def padded_bounds(self, refresh: bool = False):
        """Stage bounds stacked over the horizon: lb, ub of shape (H, G).

        G is the largest number of constraints of any stage. Missing rows are
        unbounded (+-PAD_BOUND). The result is cached; `sqp_solve` refreshes it
        at the start of every solve, in case the constraints changed.
        """
        if self._padded_bounds is not None and not refresh:
            return self._padded_bounds

        n_gs = [self.n_g(k) for k in range(self.horizon)]
        G = max(n_gs)
        lb = torch.full((self.horizon, G), -PAD_BOUND)
        ub = torch.full((self.horizon, G), PAD_BOUND)
        for k, n_g in enumerate(n_gs):
            if n_g > 0:
                lb_k, ub_k = self.g_bounds(k)
                lb, ub = lb.to(lb_k.device), ub.to(ub_k.device)
                lb[k, :n_g], ub[k, :n_g] = lb_k, ub_k

        self._padded_bounds = (lb, ub)
        return self._padded_bounds

    def padded_g(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Stage constraint values stacked over the horizon: (B, H, G)."""
        lb, _ = self.padded_bounds()
        g = torch.zeros(
            (x.shape[0], self.horizon, lb.shape[-1]), dtype=x.dtype, device=x.device
        )
        for k in range(self.horizon):
            n_g = self.n_g(k)
            if n_g == 0:
                continue
            if k < self.horizon - 1:
                g[:, k, :n_g] = self.g(k, x[:, k], u[:, k])
            else:
                g[:, k, :n_g] = self.g(k, x[:, k])
        return g

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

    def evaluate_directional_derivatives(
        self, guess: SqpSolution, correction: AdmmSolution, mat: QpParameters = None
    ):
        """Directional derivative of the total cost along (dx, du).

        If `mat` is the linearization at `guess`, its gradients (q, r) are
        reused instead of evaluating the cost gradients again.
        """
        if mat is not None:
            q, r = mat.q, mat.r
        else:
            q = torch.stack(
                [self.lx(k, guess.x[:, k], guess.u[:, k]) for k in range(self.horizon - 1)]
                + [self.lx(-1, guess.x[:, -1])],
                dim=1,
            )
            r = torch.stack(
                [self.lu(k, guess.x[:, k], guess.u[:, k]) for k in range(self.horizon - 1)],
                dim=1,
            )
        return torch.sum(q * correction.dx, dim=(1, 2)) + torch.sum(
            r * correction.du, dim=(1, 2)
        )

    def equality_residuals(self, x: torch.Tensor, u: torch.Tensor):
        """Dynamics gaps f(x_k, u_k) - x_{k+1}, shape (B, H-1, n_x), and
        underactuation residuals h(x_k, u_k), shape (B, H-1, n_h) or None."""
        x_flat = x[:, :-1].reshape(-1, self.n_x)
        u_flat = u.reshape(-1, self.n_u)
        gaps = self.dynamics.f(x_flat, u_flat, self.dt).reshape(x[:, 1:].shape) - x[:, 1:]
        h = None
        if self.underactuation is not None:
            h = self.underactuation.h(x_flat, u_flat).reshape(*u.shape[:2], self.n_h)
        return gaps, h

    def evaluate_guess(self, solution_guess: SqpSolution):
        """Return total trajectory cost and constraint violation"""
        batch_size = self.batch_size
        horizon = self.horizon
        dt = self.dt
        n_x, n_u = self.n_x, self.n_u
        x, u = solution_guess.x, solution_guess.u

        # Total trajectory cost
        cost = torch.zeros((batch_size), device=x.device)
        for k in range(horizon - 1):
            cost = cost + self.l(k, x[:, k], u[:, k])
        cost = cost + self.l(-1, x[:, -1])

        # Stage constraint violation and complementarity, all stages at once
        constr_viols_inf = torch.zeros((batch_size), device=x.device)
        comp_viols_inf = torch.zeros((batch_size), device=x.device)
        lb, ub = self.padded_bounds()
        if lb.shape[-1] > 0:
            g_val = self.padded_g(x, u)
            stage_error = torch.maximum(relu(lb - g_val), relu(g_val - ub))
            constr_viols_inf = stage_error.amax(dim=(1, 2))

            ksi = solution_guess.ksi
            if torch.is_tensor(ksi):
                # Complementarity: dual * slack = 0
                # For upper bound: max(ksi, 0) * (ub - g)
                # For lower bound: min(ksi, 0) * (g - lb)
                comp_ub = torch.abs(relu(ksi) * (ub - g_val))
                comp_lb = torch.abs(relu(-ksi) * (g_val - lb))
                comp_viols_inf = torch.maximum(comp_ub, comp_lb).amax(dim=(1, 2))

        # Dynamics violation
        x_flat = x[:, :-1].reshape(-1, n_x)
        u_flat = u.reshape(-1, n_u)
        x_pred = self.dynamics.f(x_flat, u_flat, dt).reshape(x[:, 1:].shape)
        dyn_viols_inf = torch.abs(x[:, 1:] - x_pred).amax(dim=(1, 2))

        # Underactuation violation
        if self.underactuation is not None:
            uact_violation = self.underactuation.h(x_flat, u_flat)
            max_uact_violation = (
                torch.abs(uact_violation).reshape(batch_size, -1).amax(dim=1)
            )
            constr_viols_inf = torch.maximum(constr_viols_inf, max_uact_violation)

        return cost, dyn_viols_inf, constr_viols_inf, comp_viols_inf

    def linearize(self, solution_guess: SqpSolution):
        batch_size = self.batch_size
        horizon = self.horizon
        n_x, n_u = self.n_x, self.n_u
        x, u = solution_guess.x, solution_guess.u

        Q = torch.zeros((batch_size, horizon, n_x, n_x))
        q = torch.zeros((batch_size, horizon, n_x))
        R = torch.zeros((batch_size, horizon - 1, n_u, n_u))
        r = torch.zeros((batch_size, horizon - 1, n_u))
        S = torch.zeros((batch_size, horizon - 1, n_u, n_x))

        # Dynamics and underactuation do not depend on the stage, so evaluate
        # them for all stages at once on the flattened (B * (H-1)) batch
        x_flat = x[:, :-1].reshape(-1, n_x)
        u_flat = u.reshape(-1, n_u)
        stages = (batch_size, horizon - 1)
        A = self.dynamics.fx(x_flat, u_flat, self.dt).reshape(*stages, n_x, n_x)
        B = self.dynamics.fu(x_flat, u_flat, self.dt).reshape(*stages, n_x, n_u)
        b = self.dynamics.f(x_flat, u_flat, self.dt).reshape(*stages, n_x) - x[:, 1:]

        C = None
        D = None
        d = None
        if self.underactuation is not None:
            n_h = self.n_h
            C = self.underactuation.hx(x_flat, u_flat).reshape(*stages, n_h, n_x)
            D = self.underactuation.hu(x_flat, u_flat).reshape(*stages, n_h, n_u)
            d = self.underactuation.h(x_flat, u_flat).reshape(*stages, n_h)

        # Stage constraints, padded to the largest stage dimension G
        lb, ub = self.padded_bounds()
        G = lb.shape[-1]
        M = torch.zeros((batch_size, horizon, G, n_x))
        N = torch.zeros((batch_size, horizon - 1, G, n_u))
        n = torch.zeros((batch_size, horizon, G))

        # Costs and constraints may differ per stage
        for k in range(horizon - 1):
            x_lin, u_lin = x[:, k], u[:, k]

            Q[:, k] = self.lxx(k, x_lin, u_lin)
            q[:, k] = self.lx(k, x_lin, u_lin)
            R[:, k] = self.luu(k, x_lin, u_lin)
            r[:, k] = self.lu(k, x_lin, u_lin)
            S[:, k] = self.lux(k, x_lin, u_lin)

            n_g = self.n_g(k)
            if n_g > 0:
                M[:, k, :n_g] = self.gx(k, x_lin, u_lin)
                N[:, k, :n_g] = self.gu(k, x_lin, u_lin)
                n[:, k, :n_g] = self.g(k, x_lin, u_lin)

        x_F = x[:, -1]
        Q[:, -1] = self.lxx(-1, x_F)
        q[:, -1] = self.lx(-1, x_F)

        n_g = self.n_g(horizon - 1)
        if n_g > 0:
            M[:, -1, :n_g] = self.gx(-1, x_F)
            n[:, -1, :n_g] = self.g(-1, x_F)

        return QpParameters(
            Q=Q, q=q, R=R, r=r, S=S, A=A, B=B, b=b, C=C, D=D, d=d, M=M, N=N, n=n,
            lb=lb, ub=ub,
        )
