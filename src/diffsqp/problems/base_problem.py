from abc import ABC, abstractmethod
from typing import List, Optional
import torch

from diffsqp.costs import Cost
from diffsqp.dynamics import Dynamics
from diffsqp.constraints import UnderactuationConstraint, GenericConstraint

from diffsqp.utils.math import mv


class ProblemParams:
    def __init__(self, **args):
        # self.system: str = args["name"]
        self.inverse_dynamics: bool = args["inverse_dynamics"]
        self.n_batch: int = args["n_batch"]
        self.tf: float = args["tf"]
        self.dt: float = args["dt"]
        self.n_x: int = len(args["q_w"])
        self.n_u: int = len(args["r_w"])
        self.horizon = int(self.tf / self.dt)
        # # Initial and final states
        self.x_init = torch.tensor(args["x_init"]).repeat(self.n_batch, 1)
        # Apply noise only to the first two dimensions (usually positions)
        self.x_init[:, 0:2] += args["noise_std"] * torch.randn((self.n_batch, 2))
        self.x_des = torch.tensor(args["x_des"]).repeat(self.n_batch, 1)

        # # Cost weights
        self.q_w = torch.tensor(args["q_w"])
        self.r_w = torch.tensor(args["r_w"])
        self.qf_w = torch.tensor(args["qf_w"])


class Problem(ABC):
    """
    An abstract base class representing a Trajectory Optimization problem.

    Attributes:
        horizon (int): The total number of time steps (T).
        dt (float): The integration time step.
        n_x (int): Dimension of the state vector.
        n_u (int): Dimension of the control vector.
        costs (List[List[Cost]]): A list of length `horizon`, where each element
            is a list of Cost objects active at that stage.
        dynamics (Dynamics): The dynamics model.
        underactuation (Constraints): The underactuation equality constraints.
        constraints (List[Constraints]): A list of constraint objects for each stage.
        states (List[torch.Tensor]): The current state trajectory [nB x n_x].
        controls (List[torch.Tensor]): The current control trajectory [nB x n_u].
        pi (List[torch.Tensor]): Lagrange multipliers for dynamics (equality) constraints.
        ni (List[torch.Tensor]): Lagrange multipliers for general equality constraints.
    """

    def __init__(self, params: ProblemParams) -> None:
        """
        Initializes the optimization problem buffers.

        Args:
            horizon (int): Horizon length.
            dt (float): Integration dt.
            nB (int): Batch size for parallel trajectory optimization.
            n_x (int): State dimension.
            n_u (int): Control dimension.
        """
        self.horizon = params.horizon
        self.dt = params.dt
        self.n_x = params.n_x
        self.n_u = params.n_u
        self.n_batch = params.n_batch
        self.costs: List[List[Cost]] = []
        self.dynamics: Dynamics = None
        self.underactuation: UnderactuationConstraint = None
        self.constraints: List[GenericConstraint] = [None] * self.horizon
        self.states: List[torch.Tensor] = [
            torch.zeros((self.n_batch, self.n_x)) for _ in range(self.horizon)
        ]
        self.controls: List[torch.Tensor] = [
            torch.zeros((self.n_batch, self.n_u)) for _ in range(self.horizon - 1)
        ]
        self.mu: List[torch.Tensor] = [
            torch.zeros((self.n_batch, self.n_x)) for _ in range(self.horizon)
        ]
        self.nu: List[torch.Tensor] = [
            torch.zeros((self.n_batch, self.n_u)) for _ in range(self.horizon - 1)
        ]
        self.lam: List[torch.Tensor] = [None for _ in range(self.horizon)]

        # Initialize gradient tensors
        self.Lx = torch.zeros(
            (self.horizon, self.n_batch, self.n_x), device=self.states[0].device
        )
        self.Lu = torch.zeros(
            (self.horizon - 1, self.n_batch, self.n_u), device=self.states[0].device
        )

    # --- Lagrangian Calculation Methods ---

    def Lx_Lu(self):
        """
        Calculates the gradient of the Lagrangian with respect to states (x) and controls (u).

        Returns:
            Lx (torch.Tensor): Gradients w.r.t states for stages 0 to N. Dimensions: horizon x n_batch x nx
            Lu (torch.Tensor): Gradients w.r.t controls for stages 0 to N-1. Dimensions: horizon-1 x n_batch x nx
        """

        # Intermediate stages (k = 0 to N-1)
        for k in range(self.horizon - 1):
            x_k = self.states[k]
            u_k = self.controls[k]
            mu_k = self.mu[k]
            mu_next = self.mu[k + 1]

            # --- 1a. Fetch Gradients & Jacobians ---
            cx_k = self.lx(k, x_k, u_k)  # Cost gradient w.r.t x [n_batch, n_x]
            cu_k = self.lu(k, x_k, u_k)  # Cost gradient w.r.t u [n_batch, n_u]

            fx_k = self.dynamics.fx(
                x_k, u_k, self.dt
            )  # Dynamics Jacobian w.r.t x [n_batch, n_x, n_x]
            fu_k = self.dynamics.fu(
                x_k, u_k, self.dt
            )  # Dynamics Jacobian w.r.t u [n_batch, n_x, n_u]

            # Base Lagrangian Gradients (Cost + Dynamics)
            Lx_k = cx_k + mv(fx_k.transpose(1, 2), mu_next) - mu_k
            Lu_k = cu_k + mv(fu_k.transpose(1, 2), mu_next)

            # Underactuation Constraint Terms
            if self.underactuation is not None:
                # Assuming you have a lambda multiplier list initialized elsewhere
                nu_k = self.nu[k]
                hx_k = self.underactuation.hx(
                    x_k, u_k
                )  # Underactuation Jacobian w.r.t x [n_batch, n_h, n_x]
                hu_k = self.underactuation.hu(
                    x_k, u_k
                )  # Underactuation Jacobian w.r.t u [n_batch, n_h, n_u]

                Lx_k += mv(hx_k.transpose(1, 2), nu_k)
                Lu_k += mv(hu_k.transpose(1, 2), nu_k)

            # Inequality Constraint Terms
            if self.constraints[k] is not None:
                lam_k = self.lam[k]
                gx_k = self.gx(
                    k, x_k, u_k
                )  # Constraint Jacobian w.r.t x [n_batch, n_c, n_x]
                gu_k = self.gu(
                    k, x_k, u_k
                )  # Constraint Jacobian w.r.t u [n_batch, n_c, n_u]

                Lx_k += mv(gx_k.transpose(1, 2), lam_k)
                Lu_k += mv(gu_k.transpose(1, 2), lam_k)

            # Store computed gradients
            self.Lx[k] = Lx_k
            self.Lu[k] = Lu_k

        # 2. Terminal stage (N)
        x_N = self.states[-1]
        mu_N = self.mu[-1]

        cx_N = self.lx(-1, x_N)  # Terminal cost gradient w.r.t x [n_batch, n_x]

        # TODO: Add terminal constraint support
        self.Lx[-1] = cx_N - mu_N

        self.Lx[0] = 0.0
        return self.Lx, self.Lu

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

    def g(
        self, stage_idx: int, x: torch.Tensor, u: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Concatenate all stage constraints into a single vector.

        Returns:
            A tensor of concatenated constraints [nB x total_constraints].
        """
        constr = torch.cat([c.g(x, u) for c in self.constraints[stage_idx]], dim=1)
        return constr

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
