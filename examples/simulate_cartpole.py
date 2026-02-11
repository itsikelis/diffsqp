import torch
import numpy as np

from diffsqp.dynamics import CartPoleDynamics
from diffsqp.utils.animate import CartPoleAnimator

# 1. Setup Parameters
n_batch = 2
dt = 0.001
tf = 5.0  # Reduced time for faster testing
steps = int(tf / dt)

model = CartPoleDynamics(mc=1.0, mp=1.0, lp=0.5, grav=9.81)

# 2. Initial State
x = torch.tensor([[0.0, torch.pi, 0.0, 0.0], [0.0, torch.pi, 0.0, 0.1]])
u = torch.zeros((n_batch, 1))
u = torch.tensor([[0.0]]).repeat(n_batch, 1)

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

# fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
#
# axs[0].plot(t, states[:, 0], color="b", lw=2)
# axs[0].set_ylabel("s (metres)")
# axs[0].grid(True)
#
# axs[1].plot(t, states[:, 1], color="b", lw=2)
# axs[1].set_ylabel("$\theta$ (rad)")
# axs[1].grid(True)
#
# axs[2].plot(t, controls[:, 0], color="b", lw=2)
# axs[2].set_ylabel("$\theta$ (rad)")
# axs[2].grid(True)
#
# plt.tight_layout()
# plt.show()

# 4. Animate!
anim = CartPoleAnimator(states, model.lp, dt, n_batch)
anim.animate(step_size=50)
