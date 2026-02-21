import argparse
import json
import time
from pathlib import Path

import numpy as np
import viser
from viser.extras import ViserUrdf


def create_grid_transforms(
    num_instances: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create grid positions, rotations, and scales for mesh instances."""
    grid_size = int(np.ceil(np.sqrt(num_instances)))

    # Create grid positions.
    x = np.arange(grid_size) - (grid_size - 1) / 2
    y = np.arange(grid_size) - (grid_size - 1) / 2
    xx, yy = np.meshgrid(x, y)

    positions = np.zeros((grid_size * grid_size, 3), dtype=np.float32)
    positions[:, 0] = 0.4 * xx.flatten()
    positions[:, 1] = 0.3 * yy.flatten()
    positions[:, 2] = 0.5
    positions = positions[:num_instances]

    # All instances have identity rotation.
    rotations = np.zeros((num_instances, 4), dtype=np.float32)
    rotations[:, 0] = 1.0  # w component = 1

    # Initial scales.
    scales = np.linalg.norm(positions, axis=-1)
    scales = np.sin(scales * 1.5) * 0.5 + 1.0
    return positions, rotations, scales.astype(np.float32)


def create_robot_control_sliders(
    server: viser.ViserServer, viser_urdf: ViserUrdf
) -> tuple[list[viser.GuiInputHandle[float]], list[float]]:
    """Create slider for each joint of the robot. We also update robot model
    when slider moves."""
    slider_handles: list[viser.GuiInputHandle[float]] = []
    initial_config: list[float] = []
    for joint_name, (
        lower,
        upper,
    ) in viser_urdf.get_actuated_joint_limits().items():
        lower = lower if lower is not None else -np.pi
        upper = upper if upper is not None else np.pi
        initial_pos = 0.0 if lower < -0.1 and upper > 0.1 else (lower + upper) / 2.0
        slider = server.gui.add_slider(
            label=joint_name,
            min=lower,
            max=upper,
            step=1e-3,
            initial_value=initial_pos,
        )
        slider.on_update(  # When sliders move, we update the URDF configuration.
            lambda _: viser_urdf.update_cfg(
                np.array([slider.value for slider in slider_handles])
            )
        )
        slider_handles.append(slider)
        initial_config.append(initial_pos)
    return slider_handles, initial_config


def main():
    server = viser.ViserServer()
    server.gui.configure_theme(dark_mode=True)

    RESOURCES = Path(__file__).resolve().parent
    URDF_DIR = RESOURCES / "robots"

    urdf_path = URDF_DIR / "cartpole.urdf"

    n_robots = 64

    print("Open your browser to http://localhost:8080")
    print("Press Ctrl+C to exit")

    # Scene decoration
    # server.scene.world_axes.visible = True
    # server.scene.add_grid(
    #     "/floor",
    #     width=6.0,
    #     height=6.0,
    #     plane="xy",
    #     cell_size=0.25,
    #     section_size=1.0,
    # )

    pos, rot, scales = create_grid_transforms(n_robots)
    bases = []
    robots = []
    for i in range(n_robots):
        node_name = "/robot_" + str(i)
        robot_base = server.scene.add_frame(node_name, show_axes=False)
        viser_robot = ViserUrdf(
            server, urdf_or_path=urdf_path, root_node_name=node_name
        )
        robot_base.position = pos[i]
        bases.append(robot_base)
        robots.append(viser_robot)

    # with server.gui.add_folder("Joint position control"):
    #     (slider_handles, initial_config) = create_robot_control_sliders(
    #         server, robots[0]
    #     )
    #
    # with server.gui.add_folder("Select urdf"):
    #     urdf_dropdown = server.gui.add_dropdown(
    #         "URDF",
    #         options=[urdf_path.name],
    #         initial_value=urdf_path.name,
    #     )

    while True:
        time.sleep(10.0)

    # @joint_slider.on_update
    # def _(_) -> None:
    #     idx = min(frame_slider.value, state["N"] - 1)
    #     try:
    #         state["k"] = idx
    #         robots[0].update_cfg(np.array(state["traj"][idx]["q"]))
    #     except ValueError as e:
    #         print(f"[error] update_cfg: {e}")
    #         playing.value = False


if __name__ == "__main__":
    main()
