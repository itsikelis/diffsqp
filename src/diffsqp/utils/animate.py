import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors


class CartPoleAnimator:
    def __init__(self, states, lp, dt, nB):
        self.states = states.detach().cpu().numpy().transpose(1, 0, 2)
        # self.states = states
        self.lp = lp
        self.dt = dt
        self.nB = nB

        # UI Constants
        self.cart_width = 0.3
        self.cart_height = 0.15

        # Setup Figure
        self.fig, self.axs = plt.subplots(nB, 1, figsize=(8, 3 * nB), sharex=True)
        if nB == 1:
            self.axs = [self.axs]

        self.carts = []
        self.poles = []
        colors = list(mcolors.TABLEAU_COLORS.values())

        for i in range(nB):
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
        for i in range(self.nB):
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


class AcrobotAnimator:
    def __init__(self, states, l1, l2, dt, nB):
        """
        states: Array of shape [time, nB, 4]
                where indices 0 and 1 are theta1 and theta2.
        l1, l2: Lengths of the two links.
        dt: Time step.
        nB: Number of parallel simulations to show.
        """
        self.states = states.detach().cpu().numpy().transpose(1, 0, 2)
        self.l1 = l1
        self.l2 = l2
        self.dt = dt
        self.nB = nB

        # Setup Figure
        self.fig, self.axs = plt.subplots(nB, 1, figsize=(6, 6 * nB), sharex=True)
        if nB == 1:
            self.axs = [self.axs]

        self.lines = []
        colors = list(mcolors.TABLEAU_COLORS.values())

        # Total length for axis scaling
        L_total = (l1 + l2) * 1.1

        for i in range(nB):
            ax = self.axs[i]
            color = colors[i % len(colors)]

            ax.set_aspect("equal")
            ax.set_xlim(-L_total, L_total)
            ax.set_ylim(-L_total, L_total)
            ax.grid(True)
            ax.set_title(f"Acrobot Batch {i}")

            # 'o-' creates a line with circular markers at the joints
            (line,) = ax.plot(
                [],
                [],
                "o-",
                lw=3,
                color=color,
                markersize=10,
                markerfacecolor="black",
                markeredgecolor="black",
            )
            self.lines.append(line)

    def _init(self):
        for line in self.lines:
            line.set_data([], [])
        return self.lines

    def _update(self, frame):
        for i in range(self.nB):
            # Acrobot states are usually: [theta1, theta2, dtheta1, dtheta2]
            th1 = self.states[frame, i, 0]
            th2 = self.states[frame, i, 1]

            # Elbow position (End of link 1)
            # Note: We use sin/cos based on standard vertical downward = 0
            # or horizontal = 0. Standard Acrobot: 0 is pointing down.
            x1 = self.l1 * np.sin(th1)
            y1 = -self.l1 * np.cos(th1)

            # Wrist position (End of link 2)
            # theta2 is usually relative to theta1 in Acrobot dynamics
            x2 = x1 + self.l2 * np.sin(th1 + th2)
            y2 = y1 - self.l2 * np.cos(th1 + th2)

            self.lines[i].set_data([0, x1, x2], [0, y1, y2])

        return self.lines

    def animate(self, step_size=1):
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

    def save(self, filename="acrobot_sim.mp4", step_size=1, fps=30):
        print(f"Saving animation to {filename}...")
        writer = animation.FFMpegWriter(fps=fps, bitrate=1800)

        ani = animation.FuncAnimation(
            self.fig,
            self._update,
            frames=range(0, len(self.states), step_size),
            init_func=self._init,
            blit=True,
        )
        ani.save(filename, writer=writer)
        print("Save complete.")
