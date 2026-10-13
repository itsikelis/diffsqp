from typing import NamedTuple

import torch

from diffsqp.problems import Problem
from diffsqp.types import AdmmSolution
from diffsqp.utils.math import bmm, bmv, bmtv, stage_major

MAX_REG_RETRIES = 10


class AffineRecursion:
    """x_{k+1} = T_k x_k + e_k for k = 0..n-1, with fixed matrices T_k.

    The matrices are set once per factorization and the recursion is applied
    to many right-hand sides e (one per ADMM iteration). Two schemes:

    - sequential: n batched matrix-vector products, one after the other.
    - parallel: Hillis-Steele scan over the affine maps (T_k, e_k). The
      matrix products of every scan level only depend on T, so they are
      computed here once; each `apply` then takes ceil(log2 n) vectorized
      steps instead of n. Costs ceil(log2 n) copies of T in memory.
    """

    def __init__(self, T: torch.Tensor, parallel: bool):
        self.parallel = parallel
        self.T0 = T[:, 0]
        self.n = T.shape[1]
        if parallel:
            self.levels = []
            s = 1
            while s < self.n:
                self.levels.append(T)
                T = torch.cat([T[:, :s], bmm(T[:, s:], T[:, :-s])], dim=1)
                s *= 2
        else:
            self.steps = stage_major(T)

    def apply(self, e: torch.Tensor, x0: torch.Tensor = None) -> torch.Tensor:
        """Return x_0..x_n as (B, n+1, dim); x_0 defaults to zero."""
        if x0 is not None:
            e = torch.cat([e[:, :1] + bmv(self.T0, x0).unsqueeze(1), e[:, 1:]], dim=1)
        else:
            x0 = torch.zeros_like(e[:, 0])

        if self.parallel:
            s = 1
            for T in self.levels:
                e = torch.cat([e[:, :s], e[:, s:] + bmv(T[:, s:], e[:, :-s])], dim=1)
                s *= 2
            xs = e
        else:
            es = e.unsqueeze(-1).unbind(1)
            x = es[0]
            xs = [x]
            for i in range(1, self.n):
                x = torch.baddbmm(es[i], self.steps[i], x)
                xs.append(x)
            xs = torch.stack(xs, dim=1).squeeze(-1)

        return torch.cat([x0.unsqueeze(1), xs], dim=1)


class LqrFactorization(NamedTuple):
    """Riccati quantities that depend only on the quadratic terms.

    Inside ADMM only the linear terms (q, r) change between iterations, so the
    factorization is reused until rho (and hence Q, R, S) changes.
    """

    K: torch.Tensor  # (B, H-1, n_u+n_h, n_x) feedback gains [K_u; K_nu]
    W: torch.Tensor  # (B, H-1, n_u+n_h, n_u+n_h) = -R_ext^{-1}
    P: torch.Tensor  # (B, H, n_x, n_x) value function Hessians
    Phi: torch.Tensor  # (B, H-1, n_x, n_x) closed loop A + B K_u
    c0: torch.Tensor  # (B, H-1, n_x) part of the p recursion from b and d
    Pb: torch.Tensor  # (B, H-1, n_x) P_{k+1} b_k
    forward: AffineRecursion  # dx_{k+1} = Phi_k dx_k + e_k
    backward: AffineRecursion  # p_k = Phi_k^T p_{k+1} + c_k, in reversed order
    A: torch.Tensor
    B: torch.Tensor
    b: torch.Tensor
    d: torch.Tensor
    success: torch.Tensor  # (B,) False where the factorization failed


def lqr_solve(problem: Problem, mat, reg):
    factorization = lqr_factorize(problem, mat, reg)
    return lqr_solve_factorized(factorization, mat.q, mat.r)


def lqr_factorize(
    problem: Problem, mat, reg, Q=None, R=None, S=None, parallel_scan=False
):
    """Backward Riccati recursion for the gains and value function Hessians.

    Q, R, S default to the ones in `mat` (ADMM passes its augmented versions).
    `parallel_scan` selects the scheme for the linear recursions of the solve
    (see `AffineRecursion`).
    On a failed factorization, the regularization of the failing batch
    elements is multiplied by 10 (in place, so it persists for the caller) and
    the pass is repeated. Elements that still fail are flagged in `success`
    instead of raising, so they cannot take down the whole batch.
    """
    Q = mat.Q if Q is None else Q
    R = mat.R if R is None else R
    S = mat.S if S is None else S

    for _ in range(MAX_REG_RETRIES):
        factorization = _riccati_backward(problem, reg, Q, R, S, mat, parallel_scan)
        # Single GPU->CPU sync per pass
        failed = ~factorization.success
        if not failed.any():
            return factorization
        reg[failed] *= 10.0

    return factorization


def _riccati_backward(problem: Problem, reg, Q, R, S, mat, parallel_scan):
    horizon = problem.horizon
    n_u = R.shape[-1]
    n_h = mat.D.shape[-2] if mat.C is not None else 0

    A, B = mat.A, mat.B
    # Per-stage contiguous (B, ...) blocks: strided slices make cuBLAS slow
    Q_s, R_s, S_s = stage_major(Q[:, :-1]), stage_major(R), stage_major(S)
    A_s, B_s = stage_major(A), stage_major(B)
    AT_s, BT_s = stage_major(A.mT), stage_major(B.mT)
    if n_h > 0:
        C_s, D_s, DT_s = stage_major(mat.C), stage_major(mat.D), stage_major(mat.D.mT)

    reg_I = reg.view(-1, 1, 1) * torch.eye(n_u, dtype=R.dtype, device=R.device)
    if n_h > 0:
        eye_ext = torch.eye(n_u + n_h, dtype=R.dtype, device=R.device)
        dual_reg = -1e-8 * torch.eye(n_h, dtype=R.dtype, device=R.device)
        dual_reg = dual_reg.expand(R.shape[0], n_h, n_h)

    success = torch.ones(R.shape[0], dtype=torch.bool, device=R.device)
    Ks = [None] * (horizon - 1)
    Ws = [None] * (horizon - 1)
    Ps = [None] * horizon
    Ps[-1] = Q[:, -1]

    for i in reversed(range(horizon - 1)):
        P_next = Ps[i + 1]
        BtP = BT_s[i] @ P_next
        Q_ = Q_s[i] + AT_s[i] @ (P_next @ A_s[i])
        R_ = R_s[i] + BtP @ B_s[i] + reg_I
        S_ = S_s[i] + BtP @ A_s[i]

        # The primal block must be positive definite for a descent direction
        L, info = torch.linalg.cholesky_ex(R_)
        success &= info == 0

        if n_h > 0:
            # Saddle point system [R_ D^T; D -eps I], symmetric indefinite
            R_ext = torch.cat(
                [
                    torch.cat([R_, DT_s[i]], dim=-1),
                    torch.cat([D_s[i], dual_reg], dim=-1),
                ],
                dim=-2,
            )
            S_ext = torch.cat([S_, C_s[i]], dim=-2)
            LU, pivots, info_lu = torch.linalg.lu_factor_ex(R_ext)
            success &= info_lu == 0
            W = torch.linalg.lu_solve(LU, pivots, -eye_ext.expand_as(R_ext))
        else:
            S_ext = S_
            W = -torch.cholesky_inverse(L)

        # Column-major or transposed operands make batched cuBLAS ~10x slower
        W = W.contiguous()
        K = W @ S_ext
        P = Q_ + S_ext.mT.contiguous() @ K
        Ks[i], Ws[i] = K, W
        Ps[i] = 0.5 * (P + P.mT)

    K = torch.stack(Ks, dim=1)
    W = torch.stack(Ws, dim=1)
    P = torch.stack(Ps, dim=1)
    success &= torch.isfinite(P).flatten(1).all(dim=1)
    success &= torch.isfinite(K).flatten(1).all(dim=1)

    Phi = A + B @ K[..., :n_u, :]

    # Terms of p_k that do not depend on (q, r):
    # K_nu^T d + Phi^T P_{k+1} b
    Pb = bmv(P[:, 1:], mat.b)
    c0 = bmtv(Phi, Pb)
    if n_h > 0:
        c0 = c0 + bmtv(K[..., n_u:, :], mat.d)

    return LqrFactorization(
        K=K,
        W=W,
        P=P,
        Phi=Phi,
        c0=c0,
        Pb=Pb,
        forward=AffineRecursion(Phi, parallel_scan),
        backward=AffineRecursion(Phi.mT.flip(1), parallel_scan),
        A=A,
        B=B,
        b=mat.b,
        d=mat.d,
        success=success,
    )


def lqr_with_offsets(fact: LqrFactorization, b, d=None):
    """Same factorization, with new dynamics (b) and underactuation (d)
    offsets. Only the quantities that depend on them are recomputed."""
    n_u = fact.B.shape[-1]
    Pb = bmv(fact.P[:, 1:], b)
    c0 = bmtv(fact.Phi, Pb)
    if d is not None:
        c0 = c0 + bmtv(fact.K[..., n_u:, :], d)
    return fact._replace(Pb=Pb, c0=c0, b=b, d=d)


def lqr_solve_factorized(fact: LqrFactorization, q, r):
    """Solve the LQR for new linear terms with a cached factorization.

    With the gains fixed, both passes are linear recursions:
        p_k      = Phi_k^T p_{k+1} + q_k + K_u^T r_k + c0_k
        dx_{k+1} = Phi_k dx_k + B_k k_u,k + b_k
    so each step is a single batched matrix-vector product.
    """
    n_u = fact.B.shape[-1]
    n_h = fact.K.shape[-2] - n_u
    K_u = fact.K[..., :n_u, :]

    # Backward pass for the value function gradients
    c = q[:, :-1] + bmtv(K_u, r) + fact.c0
    p = fact.backward.apply(c.flip(1), x0=q[:, -1]).flip(1)

    # Feedforward terms
    l = fact.Pb + p[:, 1:]
    r_ = r + bmtv(fact.B, l)
    if n_h > 0:
        r_ = torch.cat([r_, fact.d], dim=-1)
    k = bmv(fact.W, r_)
    k_u = k[..., :n_u]

    # Forward rollout from dx_0 = 0
    e = bmv(fact.B, k_u) + fact.b
    dx = fact.forward.apply(e)

    du_ext = bmv(fact.K, dx[:, :-1]) + k
    du = du_ext[..., :n_u]
    nu = du_ext[..., n_u:]

    mu = torch.zeros_like(dx)
    mu[:, 1:] = bmv(fact.P[:, 1:], dx[:, 1:]) + p[:, 1:]

    # Failed factorizations take no step
    failed = ~fact.success.view(-1, 1, 1)
    return AdmmSolution(
        dx=torch.where(failed, 0.0, dx),
        du=torch.where(failed, 0.0, du),
        mu=torch.where(failed, 0.0, mu),
        nu=torch.where(failed, 0.0, nu),
        z=None,
        ksi=None,
        rho=None,
        rho_inv=None,
        rho_common=None,
    )
