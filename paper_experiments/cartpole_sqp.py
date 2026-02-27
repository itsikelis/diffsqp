import json
import argparse
import torch

from diffsqp.problems import Problem
from diffsqp.costs import LqrCost, TerminalCost
from diffsqp.dynamics import CartPoleDynamics, CartPoleInverseDynamics
from diffsqp.solvers import Lqr
from diffsqp.solvers import Ssqp


def main(args):
    # Set default device (cpu or gpu)
    torch.set_default_device(args.dev)

    if args.model == "forward":
        dyn = CartPoleDynamics(mc=1.0, mp=0.1, lp=0.5, grav=9.81)
    elif args.model == "inverse":
        dyn = CartPoleInverseDynamics(mc=1.0, mp=0.1, lp=0.5, grav=9.81)
    else:
        print("ERROR: Unsupported dynamics model. Supported are forward, inverse")
        exit()

    dt = 0.05
    tf = 5.0
    horizon = int(tf / dt)
    n_batch = args.nb
    n_state = dyn.nx
    n_ctrl = dyn.nu

    x_des = torch.tensor([0.0, torch.pi, 0.0, 0.0]).repeat(n_batch, 1)

    x_init = torch.tensor([0.0, torch.pi - 0.1, 0.0, 0.0]).repeat(
        n_batch, 1
    ) + args.std * torch.randn((n_batch, n_state))

    prob = Problem(horizon, dt, n_state, n_ctrl)

    # Set stage cost and constraints
    Q = torch.tensor([1.0, 2.0, 1.5, 1.0]) * torch.eye(n_state).repeat(n_batch, 1, 1)
    R = 5e-2 * torch.eye(n_ctrl).repeat(n_batch, 1, 1)
    Qf = 0.0 * torch.eye(n_state).repeat(n_batch, 1, 1)

    for i in range(horizon - 1):
        prob.states.append(x_init.clone())
        prob.controls.append(torch.zeros((n_batch, n_ctrl)))
        prob.costs.append(LqrCost(Q, R, x_des=x_des.clone()))
        prob.stage_dynamics.append(dyn)
    # Set terminal cost
    prob.states.append(x_des.clone())
    prob.costs.append(TerminalCost(Qf, x_des.clone()))

    # Select internal QP solver
    qp_solver = Lqr(prob)
    # Create solver object
    solver = Ssqp(prob, qp_solver)
    info = solver.solve()

    info["err_final_state"] = torch.abs(torch.square(prob.states[-1] - x_des)).tolist()

    # Save solver logs to json
    with open("log.json", "w") as f:
        json.dump(info, f, indent=4)

    # Save solution (states and controls)
    if args.save:
        torch.save(prob.states, "states.pt")
        torch.save(prob.controls, "controls.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-nb", type=int, help="Batch size")
    parser.add_argument("-model", type=str, help="Dynamics model: forward, inverse")
    parser.add_argument("-dev", type=str, help="Device to solve: cpu, cuda")
    parser.add_argument(
        "-std", type=float, help="Initial state noise standard deviation"
    )
    parser.add_argument("-save", action="store_true", help="Save solution for viz")
    main(parser.parse_args())
