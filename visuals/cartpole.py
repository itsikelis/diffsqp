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
    positions[:, 0] = 0.0 * xx.flatten()
    positions[:, 1] = 0.0 * yy.flatten()
    positions[:, 2] = 0.5
    positions = positions[:num_instances]

    # All instances have identity rotation.
    rotations = np.zeros((num_instances, 4), dtype=np.float32)
    rotations[:, 0] = 1.0  # w component = 1

    # Initial scales.
    scales = np.linalg.norm(positions, axis=-1)
    scales = np.sin(scales * 1.5) * 0.5 + 1.0
    return positions, rotations, scales.astype(np.float32)


def main(args):
    server = viser.ViserServer()
    # server.gui.configure_theme(dark_mode=True)

    BASE_DIR = Path(__file__).resolve().parent.parent
    URDF_DIR = BASE_DIR / "resources" / "robots"
    urdf_path = URDF_DIR / "cartpole.urdf"

    batch_size = args.batch_size
    q_init = np.array([0.0, 0.0])

    # server.initial_camera.position = (-0.5, 2.0, 1.5)
    server.initial_camera.position = (0.0, 0.5, 0.65)
    server.initial_camera.look_at = (0.0, 0.0, 0.5)

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

    pos, rot, scales = create_grid_transforms(batch_size)

    bases = []
    robots = []

    # Instantiate the Ghost Robot
    ghost_node_name = "/ghost"
    ghost_base = server.scene.add_frame(ghost_node_name, show_axes=False)
    ghost_robot = ViserUrdf(
        server,
        urdf_or_path=urdf_path,
        root_node_name=ghost_node_name,
        mesh_color_override=[0.0, 0.0, 0.0, 0.2],
    )
    ghost_base.position = pos[0]
    ghost_robot.update_cfg(q_init)

    for i in range(batch_size):
        # Instantiate the Real Robot
        node_name = "/robot_" + str(i)
        robot_base = server.scene.add_frame(node_name, show_axes=False)
        viser_robot = ViserUrdf(
            server, urdf_or_path=urdf_path, root_node_name=node_name
        )
        robot_base.position = pos[i]
        viser_robot.update_cfg(q_init)
        bases.append(robot_base)
        robots.append(viser_robot)

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
        x_des = x_des[:, :2].numpy()
        ghost_robot.update_cfg(x_des[0])

        horizon = states.shape[1]
        images = []
        t = 0
        while True:
            x = states[:, t, :2].numpy()
            for i in range(args.batch_size):
                # The real robot updates via the trajectory
                robots[i].update_cfg(x[i])
            t += 1

            if args.record:
                images.append(client.get_render(height=1080, width=1920))
            time.sleep(0.01)
            if t == horizon:
                break
                t = 0
                time.sleep(1.0)

        if args.record:
            print("Generating and sending GIF...")
            client.send_file_download(
                args.record + ".gif",
                iio.imwrite("<bytes>", images, extension=".gif", loop=0),
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
    parser.add_argument("-file", type=str, help="Trajectory file for playback")
    parser.add_argument("-record", type=str, help="Filename to record")
    main(parser.parse_args())
