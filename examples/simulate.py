import os
import time
import argparse
import torch
import yaml

import numpy as np

from diffsqp.problems import ProblemParameters
from diffsqp.dynamics import Dynamics, AcrobotDynamics, CartPoleDynamics
from diffsqp.dynamics import AcrobotParameters, CartPoleParameters
from diffsqp.utils.animate import AcrobotAnimator, CartPoleAnimator


def load_config(config_path):
    if not os.path.exists(config_path):
        print(f"Error: Configuration file '{config_path}' not found.")
        sys.exit(1)
    with open(config_path, "r") as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(f"Error parsing YAML file: {exc}")
            sys.exit(1)
    return data


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, help="Experiment config file")
args = parser.parse_args()

print(f"Loading problem configuration from: {args.config}")
cfg = load_config(args.config)
prob_params = ProblemParameters(**cfg["problem"])
print(f"Successfully loaded parameters. Starting solver...")

# 1. Setup Parameters
nB = cfg["problem"]["n_batch"]
# dt = cfg["problem"]["dt"]
dt = 0.001
tf = 2.0  # Reduced time for faster testing
steps = int(tf / dt)

if cfg["system"]["name"] == "acrobot":
    sys_params = AcrobotParameters(**cfg["system"])
    model = AcrobotDynamics(sys_params)
elif cfg["system"]["name"] == "cartpole":
    sys_params = CartPoleParameters(**cfg["system"])
    model = CartPoleDynamics(sys_params)

# 2. Initial State
x = prob_params.x_init
u = torch.zeros((nB, sys_params.n_u))

# 3. Storage for results
state_history = [x.clone().numpy()]
control_history = [u.clone().numpy()]
time_history = [0.0]

# 4. Simulation Loop
for i in range(steps):
    x = model.f(x, u, dt)

    state_history.append(x.clone().numpy())
    control_history.append(u.clone().numpy())
    time_history.append((i + 1) * dt)


# 5. Concatenate and Plot
states = np.array(state_history)
controls = np.array(control_history)
t = np.array(time_history)

# 4. Animate!
if cfg["system"]["name"] == "acrobot":
    anim = AcrobotAnimator(
        states,
        sys_params.l1,
        sys_params.l2,
        prob_params.dt,
        prob_params.n_batch,
    )
elif cfg["system"]["name"] == "cartpole":
    anim = CartPoleAnimator(states, sys_params.lp, prob_params.dt, prob_params.n_batch)
anim.animate(step_size=100)
