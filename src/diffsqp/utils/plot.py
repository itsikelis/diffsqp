import matplotlib.pyplot as plt


def plot_trajectories(states_tensor, controls_tensor, save_filename=None):
    # Detach and convert to numpy for the first batch
    states_np = states_tensor[0, :, :].detach().cpu().numpy()
    controls_np = controls_tensor[0, :, :].detach().cpu().numpy()

    horizon_x, n_x = states_np.shape
    horizon_u, n_u = controls_np.shape

    time_x = range(horizon_x)
    time_u = range(horizon_u)

    # Create a figure with 2 vertically stacked subplots
    # sharex=True aligns the time steps on the x-axis for both plots
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # --- Top Subplot: States ---
    for i in range(n_x):
        axs[0].plot(time_x, states_np[:, i], label=f"State $x_{{{i}}}$")

    axs[0].set_ylabel("Value")
    axs[0].set_title("State Trajectory (First Environment)")
    axs[0].legend()
    axs[0].grid(True)

    # --- Bottom Subplot: Controls ---
    for i in range(n_u):
        axs[1].plot(time_u, controls_np[:, i], label=f"Control $u_{{{i}}}$")

    axs[1].set_xlabel("Time Step $k$")
    axs[1].set_ylabel("Value")
    axs[1].set_title("Control Trajectory (First Environment)")
    axs[1].legend()
    axs[1].grid(True)

    # Adjust layout to prevent overlap and display
    plt.tight_layout()
    if save_filename is not None:
        plt.savefig(save_filename)
