import argparse
import time
import torch

from diffsqp.problems import Problem
from diffsqp.costs import LqrCost, TerminalCost
from diffsqp.dynamics import CartPoleDynamics, CartPoleInverseDynamics
from diffsqp.solvers import Lqr, QP, QPTH, QPCVXPY
from diffsqp.solvers import Ssqp


def main(args):
    # Set default device (cpu or gpu)
    torch.set_default_device(args.dev)

    if args.qp == "qpth":
        # QPTH required double precision
        torch.set_default_dtype(torch.double)

    if args.model == "forward":
        dyn = CartPoleDynamics(mc=0.5, mp=0.3, lp=0.2, grav=9.81)
    elif args.model == "inverse":
        dyn = CartPoleInverseDynamics(mc=0.5, mp=0.3, lp=0.2, grav=9.81)
    else:
        print("ERROR: Unsupported dynamics model. Supported are forward, inverse")
        exit()

    dt = 0.01
    tf = 1.0
    horizon = int(tf / dt)
    n_batch = args.nb
    n_state = dyn.nx
    n_ctrl = dyn.nu

    x_des = torch.tensor([0.0, torch.pi, 0.0, 0.0]).repeat(n_batch, 1)
    if args.task == "swingup":
        x_init = args.std * torch.randn((n_batch, n_state))
    elif args.task == "balance":
        x_init = x_des + args.std * torch.randn((n_batch, n_state))

    prob = Problem(horizon, dt, n_state, n_ctrl)

    # Set stage cost and constraints
    Q = 1e-6 * torch.eye(n_state).repeat(n_batch, 1, 1)
    R = 1e-3 * torch.eye(n_ctrl).repeat(n_batch, 1, 1)
    Qf = 1e5 * torch.eye(n_state).repeat(n_batch, 1, 1)

    for i in range(horizon - 1):
        prob.states.append(x_init.clone())
        prob.controls.append(torch.zeros((n_batch, n_ctrl)))
        prob.costs.append(LqrCost(Q, R))
        prob.stage_dynamics.append(dyn)
    # Set terminal cost
    prob.states.append(x_des)
    prob.costs.append(TerminalCost(Qf, x_des))

    # Select internal QP solver
    if args.qp == "lqr":
        qp_solver = Lqr(prob)
    elif args.qp == "kkt":
        qp_solver = QP(prob)
    elif args.qp == "qpth":
        qp_solver = QPTH(prob)
    elif args.qp == "cvxpy":
        qp_solver = QPCVXPY(prob)
    else:
        print("ERROR: Unsupported QP solver. Supported are lqr, kkt, qpth, cvxpy.")
        exit()

    # Create solver object
    solver = Ssqp(prob, qp_solver)
    info = solver.solve()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-nb", type=int, help="Batch size")
    parser.add_argument("-model", type=str, help="Dynamics model: forward, inverse")
    parser.add_argument("-qp", type=str, help="QP solver to use: lqr, kkt, qpth")
    parser.add_argument("-dev", type=str, help="Device to solve: cpu, cuda")
    parser.add_argument("-task", type=str, help="Task: swingup, stabilisation")
    parser.add_argument(
        "-std", type=float, help="Initial state noise standard deviation"
    )
    main(parser.parse_args())
