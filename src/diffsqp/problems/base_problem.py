import torch
from abc import ABC, abstractmethod

from diffsqp.costs import Cost, TerminalCost
from diffsqp.dynamics import Dynamics
from diffsqp.constraints import Constraint


class Problem(ABC):
    def __init__(self, horizon: int, dt: float, nx: int, nu: int) -> None:
        self.horizon = horizon
        self.dt = dt
        self.nx = nx
        self.nu = nu
        self.costs: List[Cost | TerminalCost] = []
        self.stage_dynamics: List[Dynamics] = []
        self.constraints: List[Constraints] = [None] * self.horizon
        self.states: List[torch.Tensor] = []
        self.controls: List[torch.Tensor] = []
        self.costates: List[torch.Tensor] = []

    # def l(stage_idx, x, u):
    #     nB = x.shape[0]
    #     total_cost = torch.zeros((nB))
    #     for c in self.costs[stage_idx]:
    #         total_cost += c.l(x, u)
    #     return total_cost
    #
    # def lx(stage_idx, x, u):
    #     nB = x.shape[0]
    #     grad = torch.zeros((nB, self.nx))
    #     for c in self.costs[stage_idx]:
    #         grad += c.lx(x, u)
    #     return grad
    #
    # def lu(stage_idx, x, u):
    #     nB = x.shape[0]
    #     grad = torch.zeros((nB, self.nu))
    #     for c in self.costs[stage_idx]:
    #         grad += c.lu(x, u)
    #     return grad
    #
    # def lxx(stage_idx, x, u):
    #     nB = x.shape[0]
    #     hessian = torch.zeros((nB, self.nx, self.nx))
    #     for c in self.costs[stage_idx]:
    #         hessian += c.lxx(x, u)
    #     return hessian
    #
    # def luu(stage_idx, x, u):
    #     nB = x.shape[0]
    #     hessian = torch.zeros((nB, self.nu, self.nu))
    #     for c in self.costs[stage_idx]:
    #         hessian += c.luu(x, u)
    #     return hessian
    #
    # def lux(stage_idx, x, u):
    #     nB = x.shape[0]
    #     hessian = torch.zeros((nB, self.nu, self.nx))
    #     for c in self.costs[stage_idx]:
    #         hessian += c.lux(x, u)
    #     return hessian
    #
    # def lxu(stage_idx, x, u):
    #     nB = x.shape[0]
    #     hessian = torch.zeros((nB, self.nx, self.nu))
    #     for c in self.costs[stage_idx]:
    #         hessian += c.lxu(x, u)
    #     return hessian

    # def state(self, stage_idx: int) -> torch.Tensor:
    #     # TODO: Add check here to see if index corresponds to horizon length
    #     start = stage_idx * (self.nx + self.nu)
    #     end = start + self.nx
    #     return self.variables[:, start:end]
    #
    # def control(self, i: int) -> torch.Tensor:
    #     # TODO: Add check here to see if index corresponds to horizon length
    #     start = i * (self.nx + self.nu) + self.nx
    #     end = start + self.nu
    #     return self.variables[:, start:end]
    #
    # def set_state(self, i: int, val: torch.Tensor) -> None:
    #     start = i * (self.nx + self.nu)
    #     end = start + self.nx
    #     self.variables[:, start:end] = val
    #
    # def set_control(self, i: int, val: torch.Tensor) -> None:
    #     start = i * (self.nx + self.nu) + self.nx
    #     end = start + self.nu
    #     self.variables[:, start:end] = val
