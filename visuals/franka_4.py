import argparse
import json
import time
import imageio.v3 as iio
from pathlib import Path
import torch
import bard

import numpy as np
import viser
from viser.extras import ViserUrdf


def create_line_transforms(
    batch_size: int, spacing=1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create line positions, rotations, and scales for the Franka instances."""
    positions = np.zeros((batch_size, 3), dtype=np.float32)
    positions[:, 1] = (np.arange(batch_size) - (batch_size - 1) / 2) * spacing
    rotations = np.zeros((batch_size, 4), dtype=np.float32)
    rotations[:, 0] = 1.0
    scales = np.ones(batch_size, dtype=np.float32)
    return positions, rotations, scales


def main(args):
    server = viser.ViserServer()
    # server.gui.configure_theme(dark_mode=True)

    BASE_DIR = Path(__file__).resolve().parent.parent
    URDF_DIR = BASE_DIR / "resources" / "robots"
    urdf_path = URDF_DIR / "fp3.urdf"
    num_joints = 7

    batch_size = args.batch_size
    spacing = args.spacing if args.spacing is not None else 1.0
    q_init = np.array([0.0, 0.0, 0.0, -np.pi / 2, 0.0, np.pi / 2, 0.0])

    # server.initial_camera.position = (-0.5, 2.0, 1.5)
    server.initial_camera.position = (2.0, 0.0, 1.0)
    server.initial_camera.look_at = (0.0, 0.0, 0.0)

    print("Open your browser to http://localhost:8080")
    print("Press Ctrl+C to exit")

    server.scene.add_grid(
        "/floor",
        width=6.0,
        height=6.0,
        plane="xy",
        cell_size=0.25,
        section_size=1.0,
    )

    pos, rot, scales = create_line_transforms(batch_size, spacing)

    bases = []
    robots = []

    ghost_bases = []
    ghost_robots = []

    for i in range(batch_size):
        # 1. Instantiate the Real Robot
        node_name = "/robot_" + str(i)
        robot_base = server.scene.add_frame(node_name, show_axes=True)
        viser_robot = ViserUrdf(
            server, urdf_or_path=urdf_path, root_node_name=node_name
        )
        robot_base.position = pos[i]
        viser_robot.update_cfg(q_init)
        bases.append(robot_base)
        robots.append(viser_robot)

        # 2. Instantiate the Ghost Robot
        ghost_node_name = "/ghost_" + str(i)
        ghost_base = server.scene.add_frame(ghost_node_name, show_axes=False)
        ghost_robot = ViserUrdf(
            server,
            urdf_or_path=urdf_path,
            root_node_name=ghost_node_name,
            mesh_color_override=[0.0, 0.0, 0.0, 0.2],
        )
        ghost_base.position = pos[i]
        ghost_robot.update_cfg(q_init)
        ghost_bases.append(ghost_base)
        ghost_robots.append(ghost_robot)

    ##############################
    ## Load and play trajectory ##
    ##############################
    init_file = args.file if args.file is not None else ""
    file_input = server.gui.add_text("Trajectory File", initial_value=init_file)
    button = server.gui.add_button("Play Trajectory")

    @button.on_click
    def _(event: viser.GuiEvent) -> None:
        client = event.client
        assert client is not None

        file_path = file_input.value
        print(f"Loading trajectory from: {file_path}")

        try:
            data = torch.load(file_path)
            states = data["x"]
            controls = data["u"]
            x_des = data["x_des"]
        except Exception as e:
            print(f"Error loading file '{file_path}': {e}")
            client.add_notification(
                "File Error",
                f'Could not load from "{file_path}".',
                color="red",
            )
            return

        # Visualize target
        x_des = x_des.numpy()
        for i in range(args.batch_size):
            ghost_robots[i].update_cfg(x_des[i])

        horizon = states.shape[1]
        images = []
        t = 0
        while True:
            x = states[:, t].numpy()
            for i in range(args.batch_size):
                # The real robot updates via the trajectory
                robots[i].update_cfg(x[i])
            t += 1
            if args.record:
                images.append(client.get_render(height=1080, width=1920))
            if t == horizon:
                break
                t = 0
                time.sleep(1.0)

        if args.record:
            print("Generating and sending GIF...")
            client.send_file_download(
                "image.gif", iio.imwrite("<bytes>", images, extension=".gif", loop=0)
            )
            print("Done!")

    while True:
        time.sleep(0.1)
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-batch_size", type=int, help="Number of robots to visualize", default=4
    )
    parser.add_argument(
        "-spacing", type=float, help="Spacing between robots", default=1.0
    )
    parser.add_argument("-file", type=str, help="Trajectory file for playback")
    parser.add_argument("-record", type=bool, help="Trajectory file for playback")
    main(parser.parse_args())
