import torch

from diffsqp.problems import Problem
from diffsqp.solvers import LqrSolver


class AdmmSolver:
    def __init__(self, prob: Problem, a: float = 1.6) -> None:
        assert a > 0.0 and a < 2.0

        self.a = a
        self.prob = prob
        self.horizon = self.prob.horizon

        nB = self.prob.variables.shape[0]
        nx = self.prob.n_state
        nu = self.prob.n_ctrl
        self.delta_x = [torch.zeros((nB, nx))] * self.prob.horizon
        self.delta_u = [torch.zeros((nB, nu))] * (self.prob.horizon - 1)

        self.lqr_solver = LqrSolver(prob)

    def step(self):
        # Get delta_x and delta_y guess from LQR
        delta_x_bar, delta_u_bar = self.lqr_solver.solve()

        for k in range(self.prob.horizon - 1):
            self.delta_x[k] = self.a * delta_x_bar[k] + (1 - self.a) * self.delta_x[k]
            self.delta_u[k] = self.a * delta_u_bar[k] + (1 - self.a) * self.delta_u[k]
        self.delta_x[-1] = self.a * delta_x_bar[-1] + (1 - self.a) * self.delta_x[-1]

    def solver(self):
        # Step
        self.step()

        # Return corrections
        return self.delta_x, self.delta_u
