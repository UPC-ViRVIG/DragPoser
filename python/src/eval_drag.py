import os
import time
import argparse
import json
import random
import numpy as np
import torch
import train
import train_temporal
import eval_metrics
import warnings
from motion_data import TestMotionData
from train_data import Train_Data
from generator_architecture import Generator_Model
from temporal_transformer import Temporal
from drag_pose import DragPose
from utils import from_root_quat
from pymotion.ops.forward_kinematics_torch import fk


def main(args):
    # Set seed
    torch.manual_seed(train.param["seed"])
    random.seed(train.param["seed"])
    np.random.seed(train.param["seed"])

    device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    # Uncomment next line to use the GPU for everything
    # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Config
    mask = None
    if args.config is not None:
        with open(args.config, "r") as config_file:
            config_data = json.load(config_file)
            mask = torch.tensor(config_data["mask"], device=device)
            weights = torch.tensor(config_data["weights"], device=device)
            enable_joint_adjustment = config_data["enable_joint_adjustment"]
            joint_adjustment_indices = tuple(config_data["joint_adjustment_indices"])
            joint_adjustment_weight = config_data["joint_adjustment_weight"]
            lambda_temporal = config_data["lambda_temporal"]
            temporal_future_window = config_data["temporal_future_window"]

    # Load BVH
    filename = os.path.basename(args.input_path)
    dir = args.input_path[: -len(filename)]
    rots, pos, parents, offsets, bvh = train.get_info_from_bvh(train.get_bvh_from_disk(dir, filename))

    # Create Models
    train_data = Train_Data(device, train.param, None)
    generator_model = Generator_Model(device, train.param, parents, train_data).to(device)
    temporal_model = Temporal(train_temporal.param, device_gpu).to(device_gpu)

    # Load Models
    generator_model_path = os.path.join(args.model_path, "generator.pt")
    means, stds = train.load_model(generator_model, generator_model_path, train_data, device)
    temporal_model_path = os.path.join(args.model_path, "temporal.pt")
    means_latent, stds_latent = train_temporal.load_model(temporal_model, temporal_model_path, device)

    # TestMotionData
    eval_dataset = TestMotionData(train.param, train.scale, device, height_indices=[0, 4, 8, 13, 17, 21])
    eval_dataset.set_means_stds(means, stds)
    eval_dataset.add_motion(offsets, pos[:, 0, :], rots, parents, bvh, filename)  # only global position
    eval_dataset.normalize()

    # Mask
    if mask is None:
        # If no config file is provided, use default values
        mask = torch.tensor(
            [
                1,  # hips        - 0
                0,  # leftUpLeg   - 1
                0,  # leftLeg     - 2
                1,  # leftFoot    - 3
                0,  # leftToe     - 4
                0,  # rightUpLeg  - 5
                0,  # rightLeg    - 6
                1,  # rightFoot   - 7
                0,  # rightToe    - 8
                0,  # spine       - 9
                0,  # chest       - 10
                0,  # upperChest  - 11
                0,  # neck        - 12
                1,  # head        - 13
                0,  # leftCollar    - 14
                0,  # leftShoulder  - 15
                0,  # leftElbow     - 16
                1,  # leftWrist     - 17
                0,  # rightCollar   - 18
                0,  # rightShoulder - 19
                0,  # rightElbow    - 20
                1,  # rightWrist    - 21
            ],
            device=device,
        )
        # First component = position, second component = rotation
        weights = torch.tensor(
            [
                [10, 10],  # hips
                [1, 0.01],  # leftUpLeg
                [1, 0.01],  # leftLeg
                [5, 0.01],  # leftFoot
                [1, 0.01],  # leftToe
                [1, 0.01],  # rightUpLeg
                [1, 0.01],  # rightLeg
                [5, 0.01],  # rightFoot
                [1, 0.01],  # rightToe
                [1, 0.01],  # spine
                [1, 0.01],  # chest
                [1, 0.01],  # upperChest
                [1, 0.01],  # neck
                [5, 0.01],  # head
                [1, 0.01],  # leftCollar
                [1, 0.01],  # leftShoulder
                [1, 0.01],  # leftElbow
                [5, 0.01],  # leftWrist
                [1, 0.01],  # rightCollar
                [1, 0.01],  # rightShoulder
                [1, 0.01],  # rightElbow
                [5, 0.01],  # rightWrist
            ],
            device=device,
        )
        # Joint adjustment
        enable_joint_adjustment = True
        joint_adjustment_indices = (0, 0)  # (joint_index, end_effector_index)
        joint_adjustment_weight = 1.0
        # Number-of-sensors-dependent parameters
        lambda_temporal = 0.02  # 0.02 for 6 tracking points, 0.125 for 4 tracking points
        temporal_future_window = 0  # 0 for 6 tracking points, 16 for 4 tracking points

    mask_indices = torch.nonzero(mask).squeeze()
    weights = weights[mask_indices]
    offsets = torch.tensor(offsets, device=device)

    # Evaluate
    drag = DragPose(generator_model, temporal_model, means_latent, stds_latent, device, device_gpu)
    norm_motion = eval_dataset.get_item(0)
    input = norm_motion["dqs"].unsqueeze(0).permute(0, 2, 1)
    global_pos = norm_motion["global_pos"]
    global_rot = norm_motion["global_rot"]
    heights = norm_motion["heights"]
    # LIMIT OF FRAMES -------------
    n_frames = input.shape[-1]
    # n_frames = min(2000, input.shape[-1]) # use this to limit the number of frames
    # -----------------------------
    initial_pose = torch.tile(input[..., 0:1], (1, 1, train.param["window_size"]))
    initial_global_pos = global_pos[..., 0:1]
    initial_global_rot = global_rot[..., 0:1]
    initial_heights = heights[0]
    drag.set_initial_pose(initial_pose, initial_global_pos, initial_global_rot, initial_heights)
    results_pose = torch.zeros(
        input.reshape(input.shape[0], -1, 8, input.shape[-1])[..., :4, :].flatten(1, 2).shape[:-1]
        + (n_frames,)
    )
    results_global_pos = torch.zeros(results_pose.shape[0], 3, n_frames)
    start_time = time.time()
    for i in range(0, n_frames):
        # print("Frame {}".format(i + 1))
        if i % 100 == 0:
            print("Frame: {} out of {}".format(i + 1, n_frames))

        target_pose = input[..., i : i + 1]
        target_global_pos = global_pos[..., i : i + 1]
        target_global_rot = global_rot[..., i : i + 1]

        # get only quaternions from dual quaternions (first 4 components)
        target_qs = (
            target_pose.clone()
            .reshape((target_pose.shape[0], -1, 8, target_pose.shape[-1]))[..., :4, :]
            .flatten(1, 2)
        )
        # denormalize
        target_qs = target_qs * drag.stds_dqs + drag.means_dqs
        # change rotation of root to global rotation
        target_qs[:, :4, :] = target_global_rot
        # change joints from root space to local space
        target_qs_local = from_root_quat(
            target_qs.permute(0, 2, 1).reshape(
                (target_qs.shape[0], target_qs.shape[-1], -1, drag.channels_per_joint)
            ),
            drag.parents,
        )

        target_world_displacement = (target_global_pos - drag.current_global_pos.detach().clone()).permute(
            0, 2, 1
        )

        pos_target_qs, rotmats_target_qs = fk(
            target_qs_local,
            target_world_displacement,
            offsets,
            drag.parents,
        )

        # positions are in global space but with the root of the character at the origin
        # target_ee_pos has shape (end_effectors, 3)
        target_ee_pos = pos_target_qs[0, 0, mask_indices, :]
        # rotations are in global space
        # target_ee_rot has shape (end_effectors, 3, 3)
        target_ee_rot = rotmats_target_qs[0, 0, mask_indices, :, :]

        res_pose, res_global_pos = drag.run(
            target_ee_pos=target_ee_pos,
            target_ee_rot=target_ee_rot,
            mask_joints=mask_indices,
            weights_joints=weights,
            offsets=offsets,
            stop_eps_pos=0.01 * 0.01,
            stop_eps_rot=0.01,
            max_iter=100,
            min_loss_incr=0.00001,
            learning_rate=1e-2,
            lambda_rot=1,
            lambda_temporal=lambda_temporal,
            temporal_future_window=temporal_future_window,
            height_indices=[0, 4, 8, 13, 17, 21],
            joint_adjustment_indices=joint_adjustment_indices if enable_joint_adjustment else None,
            joint_adjustment_weight=joint_adjustment_weight,
            verbose=args.verbose,
        )
        results_pose[..., i] = res_pose
        results_global_pos[..., i] = res_global_pos

    end_time = time.time()

    # Save Result
    eval_path, eval_filename = train.result_to_bvh(
        results_pose,
        None,
        means,
        stds,
        bvh,
        filename,
        save=True,
        res_global_pos=results_global_pos,
        are_root_rot_incr=False,
    )

    # Evaluate Positional Error
    mpjpe, mpeepe = eval_metrics.eval_pos_error(
        train.get_bvh_from_disk(dir, filename),
        train.get_bvh_from_disk(eval_path, eval_filename),
        device,
        downsample_gt=1,  # change if bvh has different fps from 60
    )

    print("Evaluate Loss: {}".format(mpjpe + mpeepe))
    print("Mean Per Joint Position Error: {}".format(mpjpe))
    print("Mean End Effector Position Error: {}".format(mpeepe))
    print("Time: {}".format(end_time - start_time))


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    parser = argparse.ArgumentParser(description="Evaluate DragPoser")
    parser.add_argument(
        "model_path",
        type=str,
        help="path to pytorch model folder",
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="path to the input .bvh file or to a directory (all .bvh files in the directory will be evaluated)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="path to the config file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="print additional information",
    )
    args = parser.parse_args()

    # if input_path is a directory, then evaluate all .bvh files in that directory
    if os.path.isdir(args.input_path):
        directory = args.input_path
        for filename in os.listdir(directory):
            if filename.endswith(".bvh"):
                args.input_path = os.path.join(directory, filename)
                print("Evaluate {} ------------------------".format(args.input_path))
                main(args)
    else:
        main(args)
