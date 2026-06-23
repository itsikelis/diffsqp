from abc import ABC, abstractmethod
from typing import List, Optional
import torch

from diffsqp.costs import Cost
from diffsqp.dynamics import Dynamics
from diffsqp.constraints import UnderactuationConstraint, GenericConstraint

from diffsqp.utils.math import mv


class ProblemParameters:
    def __init__(self, **args):
        # self.system: str = args["name"]
        self.inverse_dynamics: bool = args["inverse_dynamics"]
        self.n_batch: int = args["n_batch"]
        self.tf: float = args["tf"]
        self.dt: float = args["dt"]
        self.n_x: int = len(args["q_w"])
        self.n_u: int = len(args["r_w"])
        # Number of underactuated DoFs
        self.n_h: int = args["n_h"]
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
        self.constraints: List[GenericConstraint] = [None] * self.horizon

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
