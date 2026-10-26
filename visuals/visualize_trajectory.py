"""Play back batched trajectories saved by `diffsqp.utils.load_save.save_solution`.

Each batch element is drawn as a separate robot on a grid. The trajectory file
holds `x` with shape (batch, horizon, n_x) and, optionally, `x_des`.

State layout:
  * fixed base (default): x[:n_joints] are the actuated joint positions
    (cartpole, acrobot, franka).
  * --floating_base: x[0:3] is the base position, x[3:7] the base quaternion
    (--quat_order, wxyz by default), followed by any joint positions
    (quadrotor, go2). A `floating` root joint in the URDF is removed, since the
    base pose comes from the state.

    uv run visuals/visualize_trajectory.py --urdf cartpole.urdf --file out/cartpole.pt --batch_size 16
    uv run visuals/visualize_trajectory.py --urdf quadrotor.urdf --floating_base --file out/quad.pt
    uv run visuals/visualize_trajectory.py --urdf go2.urdf --floating_base --quat_order xyzw --file out/go2.pt --batch_size 8 --dt 0.02
"""

import argparse
import io
import threading
import time
import xml.etree.ElementTree as ET
from functools import partial
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
import viser
import yourdfpy
from viser.extras import ViserUrdf


def resolve_urdf_path(filename):
    """Resolve a URDF name relative to resources/robots, or use it as a path."""
    path = Path(filename)
    if path.is_file():
        return path.resolve()
    base_dir = Path(__file__).resolve().parent.parent
    return base_dir / "resources" / "robots" / filename


def load_urdf(urdf_path, floating_base):
    """The URDF for ViserUrdf. With a floating base, `floating` joints and the
    root links they leave without children are removed, so the actuated joints
    are only the ones after the base (the base pose comes from the state)."""
    if not floating_base:
        return urdf_path
    root = ET.parse(urdf_path).getroot()
    floating = [j for j in root.findall("joint") if j.get("type") == "floating"]
    if not floating:
        return urdf_path
    for joint in floating:
        parent = joint.find("parent").get("link")
        root.remove(joint)
        if not any(
            j.find("parent").get("link") == parent for j in root.findall("joint")
        ):
            root.remove(
                next(l for l in root.findall("link") if l.get("name") == parent)
            )
    return yourdfpy.URDF.load(
        io.StringIO(ET.tostring(root, encoding="unicode")),
        filename_handler=partial(yourdfpy.filename_handler_magic, dir=urdf_path.parent),
    )


def compute_grid_transforms(batch_size, spacing_x, spacing_y, n_rows, n_cols):
    """Positions of `batch_size` instances on a grid centred at the origin."""
    if n_rows is None or n_cols is None:
        n_rows = int(np.ceil(np.sqrt(batch_size)))
        n_cols = n_rows

    if batch_size > n_rows * n_cols:
        print(
            f"Error: Batch size ({batch_size}) exceeds the proposed grid size ({n_rows} x {n_cols})."
        )
        n_rows = int(np.ceil(np.sqrt(batch_size)))
        n_cols = n_rows
        print(f"Defaulting to square grid: {n_rows} x {n_cols}.")

    x = np.arange(n_cols) - (n_cols - 1) / 2
    y = np.arange(n_rows) - (n_rows - 1) / 2
    xx, yy = np.meshgrid(x, y)

    positions = np.zeros((n_rows * n_cols, 3), dtype=np.float32)
    positions[:, 0] = xx.flatten() * spacing_x
    positions[:, 1] = yy.flatten() * spacing_y

    # Truncate to the exact number of requested instances
    return positions[:batch_size]


class RobotInstance:
    """One URDF in the scene, driven by a state vector."""

    def __init__(
        self,
        server,
        urdf_path,
        node_name,
        position,
        floating_base,
        color,
        quat_order="xyzw",
    ):
        self.base = server.scene.add_frame(node_name, show_axes=False)
        self.robot = ViserUrdf(
            server,
            urdf_or_path=load_urdf(urdf_path, floating_base),
            root_node_name=node_name,
            mesh_color_override=color,
        )
        self.offset = position
        self.floating_base = floating_base
        self.quat_order = quat_order
        self.n_joints = len(self.robot.get_actuated_joint_limits())
        self.base.position = position

    def set_state(self, x):
        joint_start = 0
        if self.floating_base:
            self.base.position = self.offset + x[0:3]
            quat = x[3:7]
            if self.quat_order == "xyzw":
                quat = np.concatenate([quat[3:], quat[:3]])
            self.base.wxyz = quat / np.linalg.norm(quat)
            joint_start = 7
        if self.n_joints > 0:
            self.robot.update_cfg(x[joint_start : joint_start + self.n_joints])

    def set_visible(self, visible):
        self.base.visible = visible


def main(args):
    server = viser.ViserServer()
    server.initial_camera.position = tuple(args.camera_pos)
    server.initial_camera.look_at = tuple(args.camera_look_at)

    server.scene.add_grid(
        "/floor",
        width=6.0,
        height=6.0,
        plane="xy",
        cell_size=0.25,
        section_size=1.0,
        position=np.array([0.0, 0.0, args.floor_z]),
    )

    urdf_path = resolve_urdf_path(args.urdf)
    positions = compute_grid_transforms(
        args.batch_size, args.spacing_x, args.spacing_y, args.n_rows, args.n_cols
    )

    robots = [
        RobotInstance(
            server,
            urdf_path,
            f"/robot_{i}",
            positions[i],
            args.floating_base,
            None,
            args.quat_order,
        )
        for i in range(args.batch_size)
    ]
    targets = []
    if args.show_target:
        targets = [
            RobotInstance(
                server,
                urdf_path,
                f"/target_{i}",
                positions[i],
                args.floating_base,
                (0.0, 0.0, 0.0, 0.2),
                args.quat_order,
            )
            for i in range(args.batch_size)
        ]

    #########################
    ## Trajectory Playback ##
    #########################
    file_input = server.gui.add_text("Trajectory File", initial_value=args.file or "")
    play_button = server.gui.add_button("Play Trajectory")
    playing = threading.Lock()

    @play_button.on_click
    def _(event: viser.GuiEvent) -> None:
        client = event.client
        assert client is not None

        if not playing.acquire(blocking=False):
            return
        try:
            file_path = file_input.value
            print(f"Loading trajectory from: {file_path}")
            try:
                data = torch.load(file_path, map_location="cpu")
                states = data["x"].numpy()
                x_des = data.get("x_des")
            except Exception as e:
                print(f"Error loading file '{file_path}': {e}")
                client.add_notification(
                    "File Error", f'Could not load from "{file_path}".', color="red"
                )
                return

            n_shown = min(args.batch_size, states.shape[0])
            for i, robot in enumerate(robots):
                robot.set_visible(i < n_shown)
            for i, target in enumerate(targets):
                target.set_visible(x_des is not None and i < n_shown)
                if x_des is not None and i < n_shown:
                    target.set_state(x_des[i].numpy())

            images = []
            for t in range(states.shape[1]):
                for i in range(n_shown):
                    robots[i].set_state(states[i, t])
                time.sleep(args.dt)
                if args.record and t % args.record_skip == 0:
                    images.append(client.get_render(height=1080, width=1920))

            if args.record:
                print("Generating and sending GIF...")
                client.send_file_download(
                    "trajectory.gif",
                    iio.imwrite("<bytes>", images, extension=".gif", loop=0),
                )
                print("Done!")
        finally:
            playing.release()

    #####################
    ## Snapshot Button ##
    #####################
    snapshot_button = server.gui.add_button("Take Snapshot")

    @snapshot_button.on_click
    def _(event: viser.GuiEvent) -> None:
        client = event.client
        if client is None:
            return

        image_pixels = client.get_render(height=1080, width=1920)
        client.send_file_download(
            "snapshot.png", iio.imwrite("<bytes>", image_pixels, extension=".png")
        )
        client.add_notification(
            "Snapshot Taken", "Image downloaded as snapshot.png", color="green"
        )

    print("Press Ctrl+C to exit")
    while True:
        time.sleep(0.1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--urdf",
        type=str,
        required=True,
        help="URDF name in resources/robots, or a path",
    )
    parser.add_argument("--file", type=str, help="Trajectory file (.pt) for playback")
    parser.add_argument(
        "--floating_base",
        action="store_true",
        help="State starts with base position + wxyz quaternion",
    )
    parser.add_argument(
        "--quat_order",
        choices=["wxyz", "xyzw"],
        default="xyzw",
        help="Base quaternion layout in the state (xyzw: gdyn / Pinocchio, e.g. go2)",
    )
    parser.add_argument(
        "--show_target", action="store_true", help="Draw translucent robots at x_des"
    )
    # Batch size and grid layout
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Number of trajectories to show"
    )
    parser.add_argument(
        "--spacing_x", type=float, default=1.0, help="X-axis grid spacing"
    )
    parser.add_argument(
        "--spacing_y", type=float, default=1.0, help="Y-axis grid spacing"
    )
    parser.add_argument("--n_rows", type=int, help="Grid rows (default: square grid)")
    parser.add_argument(
        "--n_cols", type=int, help="Grid columns (default: square grid)"
    )
    # Scene
    parser.add_argument("--floor_z", type=float, default=0.0, help="Floor height")
    parser.add_argument("--camera_pos", type=float, nargs=3, default=[2.0, 0.0, 1.0])
    parser.add_argument(
        "--camera_look_at", type=float, nargs=3, default=[0.0, 0.0, 0.0]
    )
    # Playback and recording
    parser.add_argument("--dt", type=float, default=0.01, help="Seconds between frames")
    parser.add_argument("--record", action="store_true", help="Record playback to GIF")
    parser.add_argument(
        "--record_skip", type=int, default=5, help="Record a frame every N timesteps"
    )
    main(parser.parse_args())
