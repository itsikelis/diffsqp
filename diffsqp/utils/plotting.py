import matplotlib.pyplot as plt
import torch
import numpy as np


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
