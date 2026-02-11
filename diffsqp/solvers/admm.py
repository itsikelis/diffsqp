import torch

from diffsqp.problems import Problem
from diffsqp.solvers import Lqr


class Admm:
    def __init__(self, prob: Problem, a: float = 1.6) -> None:
        assert a > 0.0 and a < 2.0

        self.a = a
        self.prob = prob
        self.horizon = self.prob.horizon

        n_batch = self.prob.states[0].shape[0]
        n_state = self.prob.n_state
        n_ctrl = self.prob.n_ctrl
        self.delta_x = [None] * self.prob.horizon
        self.delta_u = [None] * (self.prob.horizon - 1)
        self.costates = [None] * self.prob.horizon

        self.lqr_solver = Lqr(prob)

    def step(self):
        # Get delta_x and delta_y guess from LQR
        delta_x_bar, delta_u_bar = self.lqr_solver.solve()

        # Update self.delta_x, self.delta_u
        self.update_deltas(delta_x_bar, delta_u_bar)
        self.calc_costates()

    def solve(self):
        # Step
        self.step()
        # Return corrections
        return self.delta_x, self.delta_u, self.costates

    def update_deltas(self, delta_x_bar, delta_u_bar):
        ##############################################
        ## RETURNING JUST THE LQR GUESS FOR TESTING ##
        ##############################################
        self.delta_x = delta_x_bar
        self.delta_u = delta_u_bar

        # # If first, use the LQR guess only, otherwise take a linear interpolation
        # first = self.delta_x[0] is None
        # for k in range(self.prob.horizon - 1):
        #     dx = None
        #     du = None
        #     if first:
        #         dx = 1.0 * delta_x_bar[k]
        #         du = 1.0 * delta_u_bar[k]
        #     else:
        #         dx = self.a * delta_x_bar[k] + (1 - self.a) * self.delta_x[k]
        #         du = self.a * delta_u_bar[k] + (1 - self.a) * self.delta_u[k]
        #
        #     self.delta_x[k] = dx
        #     self.delta_u[k] = du
        #
        # dx_F = None
        # if first:
        #     dx_F = 2.0 * delta_x_bar[-1]
        # else:
        #     dx_F = self.a * delta_x_bar[-1] + (1 - self.a) * self.delta_x[-1]
        # self.delta_x[-1] = dx_F

    def calc_costates(self):
        self.costates = self.lqr_solver.delta_lambda
