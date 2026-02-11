import torch
import numpy as np

import matplotlib.pyplot as plt

from diffsqp.dynamics import PendulumDynamics


def plot_pendulum_trajectories(t, states, controls, title="Pendulum Simulation"):
    """
    Plots Theta, Theta_dot, and Control Inputs over time.
    t: (N,) numpy array or torch tensor
    states: (N, 2) torch tensor [theta, dtheta]
    controls: (N, 1) torch tensor [torque]
    """
    # Convert to numpy
    if isinstance(states, torch.Tensor):
        states = states.detach().cpu().numpy()
    if isinstance(controls, torch.Tensor):
        controls = controls.detach().cpu().numpy()
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()

    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # Theta (Angle)
    axs[0].plot(t, states[:, 0], color="b", lw=2)
    axs[0].set_ylabel(r"$\theta$ (rad)")
    axs[0].set_title(title)
    axs[0].grid(True)

    # Theta_dot (Angular Velocity)
    axs[1].plot(t, states[:, 1], color="g", lw=2)
    axs[1].set_ylabel(r"$\dot{\theta}$ (rad/s)")
    axs[1].grid(True)

    # Control Input (Torque)
    axs[2].step(t, controls, color="r", where="post", lw=2)
    axs[2].set_ylabel("Torque (Nm)")
    axs[2].set_xlabel("Time (s)")
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()


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
        x = model.f(x, u, dt)

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
