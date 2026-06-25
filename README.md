# diffsqp

**diffsqp** is a batchable Sequential Quadratic Programming (SQP) solver built with PyTorch. It is designed to solve trajectory optimization and optimal control problems, natively supporting both forward and inverse dynamics formulations.

## Features

* **PyTorch-Native:** Leverages PyTorch for tensor operations and GPU acceleration, allowing you to solve batches of optimization problems in parallel.
* **Modular Dynamics & Costs:** Includes built-in support for classical underactuated robotics systems like the Acrobot and CartPole. Easily specify LQR costs and underactuation constraints.
* **Forward & Inverse Dynamics:** Configure the solver to optimize over states and controls directly, or use inverse dynamics constraints depending on your problem setup.
* **Visualization:** Built-in animators for systems like CartPole and Acrobot to visualize optimized state trajectories.

## Prerequisites

This project requires **Python 3.13 or newer**.

We recommend using [uv](https://github.com/astral-sh/uv), an extremely fast Python package and project manager written in Rust, to manage dependencies and virtual environments.

## Installation

0. **Install `uv`** (if you haven't already):
   ```bash
   curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
   ```

   (For Windows or alternative installation methods, refer to the [uv documentation](https://github.com/astral-sh/uv).)

1. **Clone the repository**:
   ```bash
    git clone https://github.com/hucebot/diffsqp
    cd diffsqp
   ```

2. **Install dependencies and setup the environment**:
Because the project uses a pyproject.toml, you can use uv sync to automatically create a virtual environment and install all required dependencies (like torch, cvxpylayers, and matplotlib):
   ```bash
    uv sync
   ```
## Prerequisites

The package includes an entry-point example script at examples/example.py. This script dynamically loads problem parameters, initializes the dynamics, builds the SQP problem, and animates the resulting trajectory.

You run the examples by passing a YAML configuration file to the script.

To run an example using uv run (which automatically executes the command inside the isolated virtual environment):
   ```bash
    # Example: Run the Acrobot solver with forward dynamics
    uv run python examples/example.py -c examples/configs/config_acrobot_forward.yaml

    # Example: Run the CartPole solver with inverse dynamics
    uv run python examples/example.py -c examples/configs/config_cartpole_inverse.yaml
   ```

The script will output the solver logs to the console and, upon completion, render a visualization of the optimized trajectory.

## Project Structure

- src/diffsqp/: Core library containing the SQP solver, constraints, costs, dynamics definitions, and types.

- examples/: Demonstration scripts and YAML configuration files to run test problems.
