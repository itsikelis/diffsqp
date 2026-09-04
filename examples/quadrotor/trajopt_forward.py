import torch

from diffsqp.problems import Problem, ProblemParameters
from diffsqp.costs import LqrCost
from diffsqp.solvers import sqp_solve, SqpParameters
from diffsqp.dynamics import QuadrotorDynamics, QuadrotorParameters
from diffsqp.constraints import StateBounds, ControlBounds
from diffsqp.utils.plot import plot_trajectories
from diffsqp.utils.animate import QuadrotorAnimator
from diffsqp.types import SqpSolution

import matplotlib.pyplot as plt

# torch.set_default_device("cuda")

sqp_parameters_dict = {
    ## ADMM ##
    "admm_max_iter": 30,
    "admm_alpha": 1.6,
    "admm_sigma": 1e-6,
    # Rho related
    "admm_reset_rho": False,
    "admm_update_rho": False,
    "admm_rho_init": 0.4,
    "admm_rho_min": 1e-6,
    "admm_rho_max": 1e8,
    "admm_adaptive_rho_tolerance": 10.0,
    "admm_rho_update_iter_freq": 10,
    # Warm starting
    "admm_warm_start_unconstrained": True,
    "admm_reset_ksi": False,
    # "admm_initialize_unconstrained": False,
    # Tolerances
    "admm_abs_tolerance": 0.001,
    "admm_abs_tolerance_final": -1.0,
    "admm_rel_tolerance": 0.0001,
    "admm_rel_tolerance_final": -1.0,
    "admm_tolerance_update_steps": 0,
    ## SQP ##
    "sqp_save_solution": False,
    "sqp_warm_start": False,
    "sqp_warm_start_file_name": "lqr_solution.pt",
    "sqp_max_iter": 100,
    "merit_mu": 1e6,
    "armijo_beta": 1e-4,
    "ls_max_iter": 10,
    "sqp_eps": 1e-4,
    "qp_solver": "lqr",
    "ls_function": "merit",
}
sqp_parameters = SqpParameters(**sqp_parameters_dict)

problem_parameters_dict = {
    "inverse_dynamics": False,
    "n_h": 0,
    "batch_size": 32,
    "dt": 0.01,
    "tf": 1.0,
    "x_init": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "x_des": [1.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "noise_std": 0.01,
    # State-control bounds
    "x_lb": [
        -100.0,
        -100.0,
        -100.0,
        -1e6,
        -1e6,
        -1e6,
        -1e6,
        -1e6,
        -1e6,
        -1e6,
        -1e6,
        -1e6,
        -1e6,
    ],
    "x_ub": [
        100.0,
        100.0,
        100.0,
        1e6,
        1e6,
        1e6,
        1e6,
        1e6,
        1e6,
        1e6,
        1e6,
        1e6,
        1e6,
    ],
    "u_lb": [-1e6, -1e6, -1e6, -1e6],
    "u_ub": [1e6, 1e6, 1e6, 1e6],
    # Cost weights
    "q_w": [
        1e-8,
        1e-8,
        1e-8,
        1e-12,
        1e-12,
        1e-12,
        1e-12,
        1e-8,
        1e-8,
        1e-8,
        1e-8,
        1e-8,
        1e-8,
    ],
    "r_w": [1e-3, 1e-3, 1e-3, 1e-3],
    "qf_w": [1e5, 1e5, 1e5, 1e5, 1e5, 1e5, 1e5, 1e5, 1e5, 1e5, 1e5, 1e5, 1e5],
}
problem_parameters = ProblemParameters(**problem_parameters_dict)

system_parameters_dict = {
    "name": "quadrotor",
    "n_x": 13,  # Number of state elements
    "n_q": 7,  # Number of position elements in state
    "n_v": 6,  # Number of velocity elements in state
    "n_j": 0,  # Number of joints
    "n_u": 4,  # Number of DoF
    "mass": 0.1,
    "inertia": [0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1],
    "grav": 9.81,
}
system_parameters = QuadrotorParameters(**system_parameters_dict)

# print(sqp_parameters)
# print(problem_parameters)
# print(system_parameters)

dynamics = QuadrotorDynamics(system_parameters)

# Create problem
problem = Problem(problem_parameters)


initial_guess = SqpSolution(
    x=torch.zeros((problem.batch_size, problem.horizon, problem.n_x)),
    u=torch.zeros((problem.batch_size, problem.horizon - 1, problem.n_u)),
    mu=torch.zeros((problem.batch_size, problem.horizon, problem.n_x)),
    nu=torch.zeros((problem.batch_size, problem.horizon - 1, problem.n_h)),
    ksi=[None] * problem.horizon,
)

if sqp_parameters.sqp_warm_start:
    target_device = torch.device("cpu")
    x, u = load_solution(sqp_parameters.sqp_warm_start_file_name, device=target_device)
    intial_guess.x = x
    intial_guess.u = u
else:
    # Initialize x
    for k in range(problem.horizon - 1):
        initial_guess.x[:, k] = problem_parameters.x_init.detach().clone()
    initial_guess.x[:, -1] = problem_parameters.x_des.detach().clone()

    # Initialize u
    m = system_parameters.mass
    grav = system_parameters.grav
    initial_guess.u = torch.Tensor([m * grav, 0.0, 0.0, 0.0]).repeat(
        problem.batch_size, problem.horizon - 1, 1
    )

# Costs
Q = problem_parameters.q_w * torch.eye(dynamics.nx).repeat(
    problem_parameters.batch_size, 1, 1
)
R = problem_parameters.r_w * torch.eye(dynamics.nu).repeat(
    problem_parameters.batch_size, 1, 1
)
Qf = problem_parameters.qf_w * torch.eye(dynamics.nx).repeat(
    problem_parameters.batch_size, 1, 1
)

# Set stage costs, constraints and initial guess
for k in range(problem.horizon - 1):
    problem.costs.append([LqrCost(Q=Q, R=R)])
    problem.constraints[k] = [
        StateBounds(
            problem.n_x,
            problem.n_u,
            problem_parameters.x_lb,
            problem_parameters.x_ub,
        ),
        ControlBounds(
            problem.n_x,
            problem.n_u,
            problem_parameters.u_lb,
            problem_parameters.u_ub,
        ),
    ]
plot_trajectories(initial_guess.x, initial_guess.u)
# Terminal stage
problem.costs.append([LqrCost(Q=Qf, x_des=problem_parameters.x_des.detach().clone())])
problem.constraints[-1] = [
    StateBounds(
        problem.n_x,
        problem.n_u,
        problem_parameters.x_lb,
        problem_parameters.x_ub,
    )
]

# Dynamics Constraints
problem.dynamics = dynamics

# Solve
print(f"Solving..")
solution, log = sqp_solve(problem, sqp_parameters, initial_guess)

print(log)

plot_trajectories(solution.x, solution.u)

# Animate:
animator = QuadrotorAnimator(
    solution.x,
    problem_parameters.dt,
    problem_parameters.batch_size,
)
# animator.animate(step_size=2)
animator.save(filename="quadrotor.mp4", step_size=2)

plt.show()
