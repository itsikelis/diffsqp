import sys
import time
import torch
from diffsqp.utils.math import mm, mv

from diffsqp.problems import Problem
from diffsqp.solvers import Admm


class Sqp:
    def __init__(
        self, prob: Problem, qp_solver, max_iter=100, eps: float = 1e-4, eps_dx=1e-1
    ) -> None:
        self.max_iter = max_iter
        self.eps = eps
        self.eps_dx = eps_dx

        self.prob = prob
        self.horizon = self.prob.horizon

        self.nB = self.prob.states[0].shape[0]
        nx = self.prob.nx
        nu = self.prob.nu

        self.admm_solver = Admm(prob, qp_solver)

        # Log best line search metrics
        self.best_cost, self.best_gamma, self.best_uact = self.calc_metrics(
            self.prob.states,
            self.prob.controls,
        )

        self.terminated = torch.zeros((self.nB), dtype=torch.bool)

        self.log = {
            "nB": self.nB,
            "terminated": 0,
            "ssqp_iterations": 0,
            "t_solve_s": 0.0,
            "cuda_reserved_bytes": 0,
            "cuda_allocated_bytes": 0,
            "t_qp_solve": [0.0] * self.max_iter,
            "t_line_search": [0.0] * self.max_iter,
            "line_search_iters": [0.0] * self.max_iter,
            "max_dyn_viol": 0.0,
            "max_uact_viol": 0.0,
        }

    def solve(self):
        # Solve for max_iter steps

        print("####### SSQP Solver ########")
        t_solve_start = time.time()
        for iter in range(self.max_iter):

            # Step solver
            t_qp_solve, t_line_search, ls_iters, alpha, dones = self.step()

            # cursor up one line
            # delete last line
            # if iter > 0:
            #     sys.stdout.write("\x1b[1A")
            #     sys.stdout.write("\x1b[2K")
            #     sys.stdout.write("\x1b[1A")
            #     sys.stdout.write("\x1b[2K")
            # print("SSQP Iteration: ", iter + 1)
            # print(
            #     "Terminated Environments: ",
            #     torch.count_nonzero(self.terminated).item(),
            #     "/",
            #     self.terminated.shape[0],
            # )

            self.log["t_qp_solve"][iter] = t_qp_solve
            self.log["t_line_search"][iter] = t_line_search
            self.log["line_search_iters"][iter] = ls_iters

            # Check for terminations
            if torch.all(self.terminated):
                break
        t_solve_end = time.time()
        print("SSQP Total Iterations: ", iter + 1)
        print(
            "Terminated Environments: ",
            torch.count_nonzero(self.terminated).item(),
            "/",
            self.terminated.shape[0],
        )
        print("Solution time: ", t_solve_end - t_solve_start)

        # Fill log
        self.log["t_solve_s"] = t_solve_end - t_solve_start
        self.log["ssqp_iterations"] = iter + 1
        self.log["terminated"] = torch.count_nonzero(self.terminated).item()
        if torch.get_default_device() != "cpu":
            self.log["cuda_reserved_bytes"] = torch.cuda.memory_reserved(0)
            self.log["cuda_allocated_bytes"] = torch.cuda.memory_allocated(0)
        return self.log

    def step(self):
        delta_x, delta_u, delta_pi, delta_lam, t_qp_solve = self.admm_solver.solve()

        start = time.time()
        alpha, dones, ls_iters = self.line_search(delta_x, delta_u)
        end = time.time()
        t_line_search = end - start

        self.check_termination(delta_x, delta_u)

        return t_qp_solve, t_line_search, ls_iters, alpha, dones

    def line_search(self, delta_x, delta_u, max_iter: float = 10):
        alpha = torch.ones((self.nB, 1))
        dones = self.terminated.clone()
        i = 0
        while (not torch.all(dones)) and (i < max_iter):
            i += 1
            gamma = torch.zeros((self.nB, 1))

            # Evaluate current alpha
            x_cand, u_cand = self.calc_cadidate_solutions(
                alpha,
                self.prob.states,
                self.prob.controls,
                delta_x,
                delta_u,
            )
            cost, gamma, uact = self.calc_metrics(x_cand, u_cand)

            # Update successful environments
            cost_improved = cost < self.best_cost
            gamma_improved = gamma < self.best_gamma
            uact_improved = uact < self.best_uact
            # update_mask = (
            #     cost_improved | gamma_improved | uact_improved
            # ) & line_search_done
            update_mask = (cost_improved | gamma_improved | uact_improved) & ~dones

            # Merit Function
            # merit = cost + self.beta * gamma + self.beta * uact
            # merit_improved = merit < self.best_merit
            # update_mask = merit_improved & ~dones
            # print(merit_improved)

            if update_mask.any():
                for k in range(self.horizon - 1):
                    self.prob.states[k][update_mask] = x_cand[k][update_mask]
                    self.prob.controls[k][update_mask] = u_cand[k][update_mask]
                self.prob.states[-1][update_mask] = x_cand[-1][update_mask]
                # Mark environments as finished
                dones[update_mask] = True

            # Update best cost and gamma
            self.best_cost[cost_improved & ~dones] = cost[cost_improved & ~dones]
            self.best_gamma[gamma_improved & ~dones] = gamma[gamma_improved & ~dones]
            self.best_uact[uact_improved & ~dones] = uact[uact_improved & ~dones]

            # Update alpha for failed environments
            failed_mask = ~update_mask & ~dones
            if failed_mask.any():
                alpha[failed_mask] *= 0.5
        # if not torch.all(dones):
        #     print("Line search failed: ", dones)
        return alpha, dones, i

    def check_termination(self, delta_x, delta_u):
        ## Lagrangian Gradient ##
        # Lx, Lu = self.calc_Lx_Lu()
        # max_Lx = torch.norm(Lx, p=float("inf"), dim=1)
        # max_Lu = torch.norm(Lx, p=float("inf"), dim=1)

        max = torch.zeros(self.nB)
        for i in range(len(delta_u)):
            # cand = torch.max(torch.square(dx), dim=1).values
            dx = delta_x[i]
            du = delta_u[i]
            res = torch.max(
                mm(dx.unsqueeze(2).transpose(1, 2), dx.unsqueeze(2)).squeeze(2)
                + mm(du.unsqueeze(2).transpose(1, 2), du.unsqueeze(2)).squeeze(2),
                dim=1,
            ).values
            max[res > max] = res[res > max]
        dx = delta_x[i]
        du = delta_u[i]
        res = torch.max(
            mm(dx.unsqueeze(2).transpose(1, 2), dx.unsqueeze(2)).squeeze(2), dim=1
        ).values
        max[res > max] = res[res > max]
        dx_crit = max < self.eps_dx

        ## Dynamics Violation ##
        # Track largest violation for each batch
        max_dyn_viols = torch.zeros(self.nB)
        max_uact_viols = torch.zeros(self.nB)
        for k in range(self.horizon - 1):
            x0 = self.prob.states[k]
            u0 = self.prob.controls[k]
            x1 = self.prob.states[k + 1]
            f = self.prob.stage_dynamics[k].f

            dyn_viol = self.calc_dynamics_violation(f, x1, x0, u0)
            dyn_inf_norm = torch.norm(dyn_viol, p=float("inf"), dim=1)
            index_mask = dyn_inf_norm > max_dyn_viols
            max_dyn_viols[index_mask] = dyn_inf_norm[index_mask]

            if self.prob.constraints[k]:
                g = self.prob.constraints[k].g
                uact_viol = self.calc_underactuation_violation(g, x0, u0)
                uact_inf_norm = torch.norm(uact_viol, p=float("inf"), dim=1)
                index_mask = uact_inf_norm > max_uact_viols
                max_uact_viols[index_mask] = uact_inf_norm[index_mask]

        # terminate_Lx = max_Lx < self.eps
        # terminate_Lu = max_Lu < self.eps
        # print("Max dyn viols: ", max_dyn_viols, ", ", max_uact_viols)
        self.log["max_dyn_viol"] = torch.norm(max_dyn_viols, p=float("inf")).item()
        self.log["max_uact_viol"] = torch.norm(max_uact_viols, p=float("inf")).item()
        terminate_dyn_viols = max_dyn_viols < self.eps
        terminate_uact_viols = max_uact_viols < self.eps

        # self.terminated = torch.logical_or(terminate_Lx, terminate_Lu)
        self.terminated = torch.logical_and(terminate_dyn_viols, terminate_uact_viols)
        self.terminated = torch.logical_and(self.terminated, dx_crit)

    def calc_cadidate_solutions(self, alpha, curr_x, curr_u, delta_x, delta_u):
        x_cand = []
        u_cand = []
        for k in range(self.horizon):
            x_cand.append(curr_x[k] + torch.mul(alpha, delta_x[k]))
            if k < self.horizon - 1:
                u_cand.append(curr_u[k] + torch.mul(alpha, delta_u[k]))
        return x_cand, u_cand

    def calc_metrics(self, x_cand, u_cand):
        cost = torch.zeros((self.nB))
        gamma = torch.zeros((self.nB))
        uact = torch.zeros((self.nB))
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
            if self.prob.constraints[k]:
                uact_viol = self.calc_underactuation_violation(
                    self.prob.constraints[k].g,
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
        return g(x, u)

    # Calculate Lagrangian gradients
    def calc_Lx_Lu(self):
        dt = self.prob.dt
        Lx = torch.zeros((self.nB, self.prob.nx))
        Lu = torch.zeros((self.nB, self.prob.nu))
        for k in range(self.horizon - 1):
            x = self.prob.states[k]
            u = self.prob.controls[k]
            pi = self.pi[k]
            lam = self.lam[k]
            lx = self.prob.costs[k].lx
            lu = self.prob.costs[k].lu
            fx = self.prob.stage_dynamics[k].fx
            fu = self.prob.stage_dynamics[k].fu
            Lx += lx(x, u) + mv(torch.transpose(fx(x, u, dt), 1, 2), pi)
            Lu += lu(x, u) + mv(torch.transpose(fu(x, u, dt), 1, 2), pi)
        # Add final node cost
        x_N = self.prob.states[-1]
        lx_N = self.prob.costs[-1].lx
        Lx += lx_N(x)

        return Lx, Lu
