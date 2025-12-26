import torch
import numpy as np

# Assuming your folders have __init__.py files
from diffsqp.dynamics.pendulum import PendulumDynamics
from diffsqp.utils.plotting import plot_pendulum_trajectories


def run_simulation():
    # 1. Setup Parameters
    dt = 0.01
    tf = 5.0  # 5 seconds
    steps = int(tf / dt)

    model = PendulumDynamics(m=1.0, l=1.0, b=0.2, grav=9.81)

    # 2. Initial State: Pendulum held horizontally (pi/2)
    x = torch.tensor([[1.57, 0.0]])
    u = torch.tensor([[0.0]])  # Zero torque

    # 3. Storage for results
    state_history = [x]
    time_history = [0.0]
    control_history = [u]

    # 4. Simulation Loop
    for i in range(steps):
        # We use the RK4 method inherited from BaseDynamics
        x = model.step(x, u, dt)

        state_history.append(x)
        time_history.append((i + 1) * dt)
        control_history.append(u)

    # 5. Concatenate and Plot
    states_cat = torch.cat(state_history, dim=0)
    controls_cat = torch.cat(control_history, dim=0)
    t_array = np.array(time_history)

    plot_pendulum_trajectories(
        t_array, states_cat, controls_cat, title="Passive Pendulum Decay (RK4)"
    )


if __name__ == "__main__":
    run_simulation()
