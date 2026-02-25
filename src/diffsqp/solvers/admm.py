import time
import torch

from diffsqp.problems import Problem


class Admm:
    def __init__(self, prob: Problem, qp_solver, a: float = 1.6) -> None:
        assert a > 0.0 and a < 2.0

        self.a = a
        self.prob = prob
        self.horizon = self.prob.horizon

        n_batch = self.prob.states[0].shape[0]
        n_state = self.prob.n_state
        n_ctrl = self.prob.n_ctrl
        self.delta_x = [None] * self.prob.horizon
        self.delta_u = [None] * (self.prob.horizon - 1)
        self.delta_pi = [None] * (self.prob.horizon - 1)
        self.delta_lam = [None] * (self.prob.horizon - 1)

        self.qp_solver = qp_solver

    def step(self):
        # Get delta_x and delta_y guess from LQR
        start = time.time()
        delta_x_qp, delta_u_qp, delta_pi_qp, delta_lam_qp = self.qp_solver.solve()
        end = time.time()
        t_qp_solve = end - start

        # Update self.delta_x, self.delta_u
        self.update_deltas(delta_x_qp, delta_u_qp, delta_pi_qp, delta_lam_qp)

        # Return time elapsed for QP solve
        return t_qp_solve

    def solve(self):
        # Step
        t_qp_solve = self.step()
        # Return corrections
        return self.delta_x, self.delta_u, self.delta_pi, self.delta_lam, t_qp_solve

    def update_deltas(self, delta_x_bar, delta_u_bar, delta_pi_bar, delta_lam_bar):
        ##############################################
        ## RETURNING JUST THE LQR GUESS FOR TESTING ##
        ##############################################
        self.delta_x = delta_x_bar
        self.delta_u = delta_u_bar
        self.delta_pi = delta_pi_bar
        self.delta_lam = delta_lam_bar

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
