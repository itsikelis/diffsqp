import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors


class CartPoleAnimator:
    def __init__(self, states, lp, dt, n_batch):
        self.states = states
        self.lp = lp
        self.dt = dt
        self.n_batch = n_batch

        # UI Constants
        self.cart_width = 0.3
        self.cart_height = 0.15

        # Setup Figure
        self.fig, self.axs = plt.subplots(
            n_batch, 1, figsize=(8, 3 * n_batch), sharex=True
        )
        if n_batch == 1:
            self.axs = [self.axs]

        self.carts = []
        self.poles = []
        colors = list(mcolors.TABLEAU_COLORS.values())

        for i in range(n_batch):
            ax = self.axs[i]
            color = colors[i % len(colors)]

            ax.set_aspect("equal")
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1, 1)
            ax.grid(True)
            ax.set_title(f"Batch Instance {i}")

            rect = plt.Rectangle(
                (0, -self.cart_height / 2),
                self.cart_width,
                self.cart_height,
                fc=color,
                ec="black",
            )
            (line,) = ax.plot([], [], "o-", lw=3, color=color, markersize=8)

            ax.add_patch(rect)
            self.carts.append(rect)
            self.poles.append(line)

    def _init(self):
        for rect, line in zip(self.carts, self.poles):
            rect.set_xy((-self.cart_width / 2, -self.cart_height / 2))
            line.set_data([], [])
        return self.carts + self.poles

    def _update(self, frame):
        updated_elements = []
        for i in range(self.n_batch):
            s = self.states[frame, i, 0]
            theta = self.states[frame, i, 1]

            self.carts[i].set_x(s - self.cart_width / 2)

            # Physics: x = s + L sin(theta), y = -L cos(theta)
            pole_x = [s, s + self.lp * np.sin(theta)]
            pole_y = [0, -self.lp * np.cos(theta)]
            self.poles[i].set_data(pole_x, pole_y)

            updated_elements.extend([self.carts[i], self.poles[i]])
        return updated_elements

    def animate(self, step_size=40):
        ani = animation.FuncAnimation(
            self.fig,
            self._update,
            frames=range(0, len(self.states), step_size),
            init_func=self._init,
            blit=True,
            interval=self.dt * step_size * 1000,
            repeat=True,
        )
        plt.tight_layout()
        plt.show()

    def save(self, filename="cartpole_sim.mp4", step_size=40, fps=30):
        """
        Saves the animation to a file using FFmpeg.
        Note: Requires 'ffmpeg' installed on your system.
        """
        print(f"Saving animation to {filename}...")

        # Define the writer (FFMpegWriter is the standard for MP4)
        writer = animation.FFMpegWriter(
            fps=fps, metadata=dict(artist="Gemini-DiffSQP"), bitrate=1800
        )

        # Calculate frames based on step_size
        frame_indices = range(0, len(self.states), step_size)

        # Wrap the animation logic in the save call
        ani = animation.FuncAnimation(
            self.fig,
            self._update,
            frames=frame_indices,
            init_func=self._init,
            blit=True,
        )

        ani.save(filename, writer=writer)
        print("Save complete.")
