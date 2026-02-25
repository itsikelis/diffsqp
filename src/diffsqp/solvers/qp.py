import torch

from diffsqp.problems import Problem


class QP:
    def __init__(self, prob: Problem) -> None:
        self.prob = prob
        self.horizon = self.prob.horizon
        self.n_batch = self.prob.states[0].shape[0]
        self.nx = self.prob.n_state
        self.nu = self.prob.n_ctrl
        self.nvars = (self.horizon - 1) * (self.nx + self.nu) + self.nx

        self.Dx = [None] * self.horizon
        self.Du = [None] * (self.horizon - 1)
        # Lagrange multipliers of the actuation part
        self.Dpi = [None] * (self.horizon - 1)
        # Lagrange multipliers of the underactuation part
        self.Dlam = [None] * (self.horizon - 1)

    def solve(self):
        Q_qp, p_qp, G_qp, h_qp, A_qp, b_qp = self.generate_problem_()

        sol, _ = self.solve_kkt_(Q_qp, p_qp, A_qp, b_qp)

        for i in range(self.horizon - 1):
            start = i * (self.nx + self.nu)
            self.Dx[i] = sol[:, start : start + self.nx]
            self.Du[i] = sol[:, start + self.nx : start + self.nx + self.nu]
        self.Dx[-1] = sol[:, -self.nx :]

        # Return corrections
        return self.Dx, self.Du, self.Dpi, self.Dlam

    def generate_problem_(self):
        nB = self.n_batch
        nhor = self.horizon
        nx = self.nx
        nu = self.nu
        nvars = self.nvars
        I = torch.eye(nx).repeat(nB, 1, 1)
        Q_qp = torch.zeros((nB, nvars, nvars))
        p_qp = torch.zeros((nB, nvars))
        A_qp1 = torch.zeros((nB, (nhor - 1) * nx, nvars))
        b_qp1 = torch.zeros((nB, (nhor - 1) * nx))

        A_qp2 = None
        b_qp2 = None
        if self.prob.stage_dynamics[0].type == "inverse":
            ng = self.prob.stage_dynamics[0].ng
            A_qp2 = torch.zeros((nB, (nhor - 1) * ng, nvars))
            b_qp2 = torch.zeros((nB, (nhor - 1) * ng))
        for i in range(self.horizon - 1):
            x_lin = self.prob.states[i]
            u_lin = self.prob.controls[i]
            x_next = self.prob.states[i + 1]

            # Create cost matrices
            Q, R, S, q, r = self.calc_linearized_cost_terms_(
                x_lin, u_lin, self.prob.costs[i]
            )
            ST = torch.transpose(S, 1, 2)

            d_idx = i * (nx + nu)
            Q_qp[:, d_idx : d_idx + nx, d_idx : d_idx + nx] = Q
            Q_qp[:, d_idx : d_idx + nx, d_idx + nx : d_idx + nx + nu] = ST
            Q_qp[:, d_idx + nx : d_idx + nx + nu, d_idx : d_idx + nx] = S
            Q_qp[:, d_idx + nx : d_idx + nx + nu, d_idx + nx : d_idx + nx + nu] = R

            p_qp[:, d_idx : d_idx + nx] = q
            p_qp[:, d_idx + nx : d_idx + nx + nu] = r

            # Create constraint matrices
            A, B, b, C, D, e = self.calc_linearized_dynamic_terms_(
                x_lin, u_lin, x_next, self.prob.stage_dynamics[i]
            )
            c_i = i * (nx + nu)
            r_i = i * nx
            A_qp1[:, r_i : r_i + nx, c_i : c_i + nx] = A
            A_qp1[:, r_i : r_i + nx, c_i + nx : c_i + nx + nu] = B
            A_qp1[:, r_i : r_i + nx, c_i + nx + nu : c_i + nx + nu + nx] = -I

            b_qp1[:, r_i : r_i + nx] = -b

            if self.prob.stage_dynamics[0].type == "inverse":
                ng = self.prob.stage_dynamics[0].ng
                c_i = i * (nx + nu)
                r_i = i * ng
                A_qp2[:, r_i : r_i + ng, c_i : c_i + nx] = C
                A_qp2[:, r_i : r_i + ng, c_i + nx : c_i + nx + nu] = D
                b_qp2[:, r_i : r_i + ng] = -e

        x_F = self.prob.states[-1]
        Q_F = self.prob.costs[-1].lxx(x_F)
        q_F = self.prob.costs[-1].lx(x_F)
        Q_qp[:, -nx:, -nx:] = Q_F
        p_qp[:, -nx:] = q_F

        A_qp1[:, -nx:, -nx:] = -I

        # Initial state constraint
        Is = torch.zeros((nB, nx, nvars))
        Is[:, 0:nx, 0:nx] = torch.eye(nx)
        A_qp1 = torch.cat([A_qp1, Is], dim=1)
        b_qp1 = torch.cat([b_qp1, torch.zeros((nB, nx))], dim=1)

        A_qp = None
        b_qp = None
        if self.prob.stage_dynamics[0].type == "inverse":
            A_qp = torch.cat([A_qp1, A_qp2], dim=1)
            b_qp = torch.cat([b_qp1, b_qp2], dim=1)
        else:
            A_qp = A_qp1
            b_qp = b_qp1

        # torch.set_printoptions(precision=2, linewidth=1000)
        # print(A_qp2[0])
        # print(b_qp2[0])
        # exit()

        G_qp = torch.zeros((nB, 1, nvars))
        G_qp[:, 0, 0] = 1.0
        h_qp = 1e6 * torch.ones((nB, 1))

        # Q_qp += 10 * torch.eye(nvars).repeat(nB, 1, 1)
        # Q_qp = self.ensure_psd(Q_qp)
        return Q_qp, p_qp, G_qp, h_qp, A_qp, b_qp

    # def ensure_psd(self, Q, eps=1e-6, verbose=True):
    #     # 1. Symmetrize to fix any numerical noise (QP matrices must be symmetric)
    #     Q_sym = (Q + Q.transpose(-1, -2)) / 2
    #
    #     # 2. Compute eigenvalues and eigenvectors
    #     # torch.linalg.eigh is optimized for Hermitian/symmetric matrices
    #     L, V = torch.linalg.eigh(Q_sym)
    #
    #     # 3. Check for negative eigenvalues
    #     min_eig = L.min(dim=-1)[0]  # Min eigenvalue per batch
    #
    #     if (min_eig < -eps).any():
    #         if verbose:
    #             print(
    #                 f"Warning: Matrix not PSD. Min eigenvalue: {min_eig.min().item():.2e}. Fixing..."
    #             )
    #
    #         # 4. Clamp eigenvalues to be at least eps
    #         # This projects the matrix onto the PSD cone (closest PSD matrix in Frobenius norm)
    #         L_new = torch.clamp(L, min=eps)
    #
    #         # 5. Reconstruct the matrix
    #         Q_psd = V @ torch.diag_embed(L_new) @ V.transpose(-1, -2)
    #         return Q_psd
    #
    #     return Q_sym

    def calc_linearized_cost_terms_(self, x_lin, u_lin, c):
        Q = c.lxx(x_lin, u_lin)
        q = c.lx(x_lin, u_lin)
        R = c.luu(x_lin, u_lin)
        r = c.lu(x_lin, u_lin)
        S = c.lux(x_lin, u_lin)
        return Q, R, S, q, r

    def calc_linearized_dynamic_terms_(self, x_lin, u_lin, x_next, dynamics):
        x_pred = dynamics.f(x_lin, u_lin, self.prob.dt)
        b = x_pred - x_next
        A = dynamics.fx(x_lin, u_lin, self.prob.dt)
        B = dynamics.fu(x_lin, u_lin, self.prob.dt)
        C = None
        D = None
        e = None
        if dynamics.type == "inverse":
            C = dynamics.gx(x_lin, u_lin)
            D = dynamics.gu(x_lin, u_lin)
            e = dynamics.g(x_lin, u_lin)
        return A, B, b, C, D, e

    def solve_kkt_(self, Q, q, A, b):
        batch_size, n, _ = Q.shape
        m = A.shape[1]

        zeros = torch.zeros((batch_size, m, m), device=Q.device, dtype=Q.dtype)
        top_row = torch.cat([Q, A.transpose(1, 2)], dim=2)  # Result: (B, n, n+m)
        bottom_row = torch.cat([A, zeros], dim=2)  # Result: (B, m, n+m)
        KKT_matrix = torch.cat([top_row, bottom_row], dim=1)  # Result: (B, n+m, n+m)

        rhs = torch.cat([-q, b], dim=1)  # Result: (B, n+m)

        solution = torch.linalg.solve(KKT_matrix, rhs)

        x = solution[:, :n]
        lam = solution[:, n:]

        return x, lam
