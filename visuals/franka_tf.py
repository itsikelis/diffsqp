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


def main(args):
    server = viser.ViserServer()
    # server.gui.configure_theme(dark_mode=True)

    BASE_DIR = Path(__file__).resolve().parent.parent
    URDF_DIR = BASE_DIR / "resources" / "robots"
    urdf_path = URDF_DIR / "fp3.urdf"
    num_joints = 7

    q_init = torch.tensor([0.0, 0.0, 0.0, -np.pi / 2, 0.0, np.pi / 2, 0.0])
    q = q_init.detach().clone()

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

    # Instantiate bard model
    bard_model = bard.build_model_from_urdf(urdf_path, floating_base=False)
    bard_model.to(dtype=torch.float32, device="cpu")
    bard_data = bard.create_data(bard_model, max_batch_size=1)

    # Instantiate the real robot
    node_name = "/robot"
    robot_base = server.scene.add_frame(node_name, show_axes=True)
    robot = ViserUrdf(server, urdf_or_path=urdf_path, root_node_name=node_name)
    robot_base.position = np.zeros(3, dtype=np.float32)
    robot.update_cfg(q_init.numpy())

    ###################
    ## Joint Sliders ##
    ###################
    joint_sliders = []
    for i in range(num_joints):
        slider = server.gui.add_slider(
            f"Joint {i}",
            min=-3.14,
            max=3.14,
            step=0.01,
            initial_value=q_init[i].item(),
        )
        joint_sliders.append(slider)

    # Define a single callback to handle any slider movement
    def on_slider_update(event: viser.GuiEvent) -> None:
        q[:] = torch.tensor([s.value for s in joint_sliders])
        robot.update_cfg(q.numpy())

    # Attach the callback to every slider
    for slider in joint_sliders:
        slider.on_update(on_slider_update)

    ####################################
    ## Read and print end-effector TF ##
    ####################################
    init_file = args.file if args.file is not None else ""
    text_input = server.gui.add_text("End-Effector Name", initial_value="fp3_link7")
    button = server.gui.add_button("Get Tf")

    @button.on_click
    def _(event: viser.GuiEvent) -> None:
        client = event.client
        assert client is not None

        print(robot)

        eef_id = bard_model.get_frame_id(text_input.value)
        transforms_batch = bard.forward_kinematics(
            model=bard_model, data=bard_data, frame_id=eef_id, q=q.unsqueeze(0)
        )
        pose_matrix = transforms_batch[0]

        print("q: ", q)
        print("tf: ", pose_matrix)

    ###########################
    ## Reset robot to q_init ##
    ###########################
    reset_button = server.gui.add_button("Reset")

    @reset_button.on_click
    def _(event: viser.GuiEvent) -> None:
        q[:] = q_init.detach().clone()
        robot.update_cfg(q.numpy())
        for i, slider in enumerate(joint_sliders):
            slider.value = q_init[i].item()

    while True:
        time.sleep(0.1)
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-file", type=str, help="Trajectory file for playback")
    parser.add_argument("-record", type=bool, help="Trajectory file for playback")
    main(parser.parse_args())
