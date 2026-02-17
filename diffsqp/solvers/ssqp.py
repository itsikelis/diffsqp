import torch
from torch import bmm

from diffsqp.problems import Problem
from diffsqp.solvers import Admm


class Ssqp:
    def __init__(self, prob: Problem, eps: float = 1e-4) -> None:
        self.eps = eps

        self.prob = prob
        self.horizon = self.prob.horizon

        self.n_batch = self.prob.states[0].shape[0]
        nx = self.prob.n_state
        nu = self.prob.n_ctrl

        self.admm_solver = Admm(prob)

        # Log best line search metrics
        self.best_cost, self.best_gamma, self.best_uact = self.calc_metrics(
            self.prob.states,
            self.prob.controls,
        )

        self.terminated = torch.zeros((self.n_batch), dtype=torch.bool)

    def solve(self, max_iter=100):
        iter = 0
        while not torch.all(self.terminated):
            print("################")
            print("SSQP Iteration: ", iter)
            self.step()
            iter += 1
            if iter >= max_iter:
                break

    def step(self):
        delta_x, delta_u, delta_pi, delta_lam = self.admm_solver.solve()
        self.line_search(delta_x, delta_u)
        self.check_termination()

    def line_search(self, delta_x, delta_u, max_iter: float = 10):
        alpha = torch.ones((self.n_batch, 1))
        line_search_done = self.terminated.clone()
        i = 0
        while (not torch.all(line_search_done)) and (i < max_iter):
            # print("Line search iter: ", i)
            i += 1
            gamma = torch.zeros((self.n_batch, 1))

            # Evaluate current alpha
            x_cand, u_cand = self.calc_cadidate_solutions(
                alpha,
                self.prob.states,
                self.prob.controls,
                delta_x,
                delta_u,
            )
            cost, gamma, uact = self.calc_metrics(x_cand, u_cand)

            # Update successful batches
            cost_improved = cost < self.best_cost
            gamma_improved = gamma < self.best_gamma
            uact_improved = uact < self.best_uact
            # update_mask = (
            #     cost_improved | gamma_improved | uact_improved
            # ) & ~line_search_done
            update_mask = (cost_improved | gamma_improved) & ~line_search_done
            if update_mask.any():
                for k in range(self.horizon - 1):
                    self.prob.states[k][update_mask] = x_cand[k][update_mask]
                    self.prob.controls[k][update_mask] = u_cand[k][update_mask]
                self.prob.states[-1][update_mask] = x_cand[-1][update_mask]
                # Mark batches as finished
                line_search_done[update_mask] = True

            # Update best cost and gamma
            self.best_cost[cost_improved & ~line_search_done] = cost[
                cost_improved & ~line_search_done
            ]
            self.best_gamma[gamma_improved & ~line_search_done] = gamma[
                gamma_improved & ~line_search_done
            ]
            self.best_uact[uact_improved & ~line_search_done] = uact[
                uact_improved & ~line_search_done
            ]

            # Update alpha for failed batches
            failed_mask = ~update_mask & ~line_search_done
            if failed_mask.any():
                alpha[failed_mask] *= 0.5
        if not torch.all(line_search_done):
            print("Line search failed: ", line_search_done)

    def check_termination(self):
        ## Lagrangian Gradient ##
        # Lx, Lu = self.calc_Lx_Lu()
        # max_Lx = torch.norm(Lx, p=float("inf"), dim=1)
        # max_Lu = torch.norm(Lx, p=float("inf"), dim=1)

        ## Dynamics Violation ##
        # Track largest violation for each batch
        max_dyn_viols = torch.zeros(self.n_batch)
        max_uact_viols = torch.zeros(self.n_batch)
        for k in range(self.horizon - 1):
            dyn_viol = self.calc_dynamics_violation(
                self.prob.stage_dynamics[k].f,
                self.prob.states[k + 1],
                self.prob.states[k],
                self.prob.controls[k],
            )
            dyn_inf_norm = torch.norm(dyn_viol, p=float("inf"), dim=1)
            index_mask = dyn_inf_norm > max_dyn_viols
            max_dyn_viols[index_mask] = dyn_inf_norm[index_mask]

            if self.prob.stage_dynamics[k].type == "inverse":
                uact_viol = self.calc_underactuation_violation(
                    self.prob.stage_dynamics[k].g,
                    self.prob.states[k],
                    self.prob.controls[k],
                )
                uact_inf_norm = torch.norm(uact_viol, p=float("inf"), dim=1)
                index_mask = uact_inf_norm > max_uact_viols
                max_uact_viols[index_mask] = uact_inf_norm[index_mask]

        # terminate_Lx = max_Lx < self.eps
        # terminate_Lu = max_Lu < self.eps
        print(
            "Terminated environments: ",
            self.terminated,
            "max_dyn_viols: ",
            max_dyn_viols,
            "max_uact_viols: ",
            max_uact_viols,
        )
        terminate_dyn_viols = max_dyn_viols < self.eps
        terminate_uact_viols = max_uact_viols < self.eps

        # self.terminated = torch.logical_or(terminate_Lx, terminate_Lu)
        self.terminated = torch.logical_and(terminate_dyn_viols, terminate_uact_viols)

    def calc_cadidate_solutions(self, alpha, curr_x, curr_u, delta_x, delta_u):
        x_cand = []
        u_cand = []
        for k in range(self.horizon):
            x_cand.append(curr_x[k] + torch.mul(alpha, delta_x[k]))
            if k < self.horizon - 1:
                u_cand.append(curr_u[k] + torch.mul(alpha, delta_u[k]))
        return x_cand, u_cand

    def calc_metrics(self, x_cand, u_cand):
        cost = torch.zeros((self.n_batch))
        gamma = torch.zeros((self.n_batch))
        uact = torch.zeros((self.n_batch))
        for k in range(self.horizon - 1):
            # Calculate total trajectory cost
            cost += self.prob.costs[k].l(x_cand[k], u_cand[k])
            # Calculate constraint violations
            dyn_viol = self.calc_dynamics_violation(
                self.prob.stage_dynamics[k].f,
                x_cand[k + 1],
                x_cand[k],
                u_cand[k],
            )
            gamma += torch.norm(dyn_viol, p=float("inf"), dim=1)
            if self.prob.stage_dynamics[k].type == "inverse":
                uact_viol = self.calc_underactuation_violation(
                    self.prob.stage_dynamics[k].g,
                    x_cand[k],
                    u_cand[k],
                )
                uact += torch.norm(uact_viol, p=float("inf"), dim=1)
        # Add final node cost
        cost += self.prob.costs[-1].l(x_cand[-1])
        return cost, gamma, uact

    def calc_dynamics_violation(self, f, x_next, x, u):
        return x_next - f(x, u, self.prob.dt)

    def calc_underactuation_violation(self, g, x, u):
        # print("g(x , u) = ", g(x, u)[0])
        return g(x, u)

    # Calculate Lagrangian gradients
    def calc_Lx_Lu(self):
        dt = self.prob.dt
        Lx = torch.zeros((self.n_batch, self.prob.n_state))
        Lu = torch.zeros((self.n_batch, self.prob.n_ctrl))
        for k in range(self.horizon - 1):
            x = self.prob.states[k]
            u = self.prob.controls[k]
            lagr = self.prob.costates[k + 1]
            lx = self.prob.costs[k].lx
            lu = self.prob.costs[k].lu
            fx = self.prob.stage_dynamics[k].fx
            fu = self.prob.stage_dynamics[k].fu
            Lx += lx(x, u) + bmm(
                torch.transpose(fx(x, u, dt), 1, 2), lagr.unsqueeze(2)
            ).squeeze(2)
            Lu += lu(x, u) + bmm(
                torch.transpose(fu(x, u, dt), 1, 2), lagr.unsqueeze(2)
            ).squeeze(2)
        # Add final node cost
        x_N = self.prob.states[-1]
        lx_N = self.prob.costs[-1].lx
        Lx += lx_N(x)

        return Lx, Lu
