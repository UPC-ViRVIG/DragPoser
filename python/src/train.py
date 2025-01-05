import random
import torch
import warnings
import time
import argparse
import os
import eval_metrics
import numpy as np
import pymotion.rotations.quat as quat
from torch.utils.data.dataloader import DataLoader
from motion_data import TestMotionData, TrainMotionData
from pymotion.io.bvh import BVH
from train_data import Train_Data
from generator_architecture import Generator_Model

scale = 1

param = {
    "batch_size": 64,
    "epochs": 1500,
    "kernel_size_temporal_dim": 1,
    "neighbor_distance": 2,
    "stride_encoder_conv": 1,
    "channel_factor": 1,
    "learning_rate": 1e-4,
    "clip_grad_value": 100.0,
    "lambda_root": 1,
    "lambda_kld": 0.001,
    "lambda_displacement": 10,
    "lambda_consecutive": 1,
    "lambda_fk": 100,
    "window_size": 1,
    "window_step": 1,
    "seed": 2222,
    "sparse_joints": [  # only used for evaluation
        0,  # first should be root (as assumed by loss.py)
        4,  # left foot
        8,  # right foot
        13,  # head
        17,  # left hand
        21,  # right hand
    ],
    "latent_dim": 24,
    "downsample": 1,
}

assert param["kernel_size_temporal_dim"] % 2 == 1


def main(args):
    # Set seed
    torch.manual_seed(param["seed"])
    random.seed(param["seed"])
    np.random.seed(param["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    # Additional Info when using cuda
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))

    # Prepare Data
    train_eval_dir = args.data_path
    # check if train and eval directories exist
    train_dir = os.path.join(train_eval_dir, "train")
    if not os.path.exists(train_dir):
        raise ValueError("train directory does not exist")
    train_files = os.listdir(train_dir)
    eval_dir = os.path.join(train_eval_dir, "eval")
    if not os.path.exists(eval_dir):
        raise ValueError("eval directory does not exist")
    eval_files = os.listdir(eval_dir)
    train_dataset = TrainMotionData(param, scale, train_eval_dir, device)
    eval_dataset = TestMotionData(param, scale, device)
    reference_parents = None  # used to make sure all bvh have the same structure
    # Train Files
    for filename in train_files:
        if filename[-4:] == ".bvh":
            rots, pos, parents, offsets, _ = get_info_from_bvh(get_bvh_from_disk(train_dir, filename))
            if reference_parents is None:
                reference_parents = parents.copy()
            assert np.all(reference_parents == parents)  # make sure all bvh have the same structure
            # Train Dataset
            train_dataset.add_motion(
                offsets,
                pos[:, 0, :],  # only global position
                rots,
                parents,
            )
    # Once all train files are added, compute the means and stds and normalize
    train_dataset.normalize()
    eval_dataset.set_means_stds(train_dataset.means, train_dataset.stds)
    # Eval Files
    for filename in eval_files:
        if filename[-4:] == ".bvh":
            rots, pos, parents, offsets, bvh = get_info_from_bvh(get_bvh_from_disk(eval_dir, filename))
            assert np.all(reference_parents == parents)  # make sure all bvh have the same structure
            # Eval Dataset
            eval_dataset.add_motion(
                offsets,
                pos[:, 0, :],  # only global position
                rots,
                parents,
                bvh,
                filename,
            )
    # Once all eval files are added, normalize
    eval_dataset.normalize()

    # Create Model
    train_data = Train_Data(device, param, args)
    generator_model = Generator_Model(device, param, reference_parents, train_data).to(device)
    train_data.set_means(train_dataset.means["dqs"], train_dataset.means["displacement"])
    train_data.set_stds(train_dataset.stds["dqs"], train_dataset.stds["displacement"])

    # Dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=param["batch_size"], shuffle=True)

    # Load Model
    best_evaluation = float("inf")
    _, generator_path, _, _ = get_model_paths(args.name, train_eval_dir)
    if args.load:
        load_model(generator_model, generator_path, train_data, device)
        # Check previous best evaluation loss
        results = evaluate_generator(generator_model, train_data, eval_dataset)
        mpjpe, mpeepe = eval_save_result(
            results,
            train_dataset.means,
            train_dataset.stds,
            eval_dir,
            device,
            save=False,
        )
        best_evaluation = mpjpe + mpeepe
    # Training Loop
    start_time = time.time()
    for epoch in range(param["epochs"]):
        avg_train_loss = 0.0
        avg_train_losses = None
        epoch_time = time.time()
        for step, (
            (motion, norm_motion),
            (_, next_norm_motion),
        ) in enumerate(train_dataloader):
            # Forward
            train_data.set_motions(
                motion["offsets"],
                norm_motion["dqs"],
                norm_motion["displacement"],
                next_norm_motion["dqs"],
                next_norm_motion["displacement"],
            )
            generator_model.train()
            generator_model.forward()
            # Loss
            loss_generator, losses_generator = generator_model.optimize_parameters()
            loss = loss_generator.item()
            avg_train_loss += loss
            if avg_train_losses is None:
                avg_train_losses = list(losses_generator)
            else:
                for i, (name, l) in enumerate(losses_generator):
                    avg_train_losses[i] = (name, avg_train_losses[i][1] + l)
            # Evaluate & Print
            if step == len(train_dataloader) - 1:
                results = evaluate_generator(generator_model, train_data, eval_dataset)
                mpjpe, mpeepe = eval_save_result(
                    results,
                    train_dataset.means,
                    train_dataset.stds,
                    eval_dir,
                    device,
                    save=False,
                )
                evaluation_loss = mpjpe + mpeepe
                # If best, save model
                was_best = False
                if evaluation_loss < best_evaluation:
                    save_model(
                        generator_model,
                        None,
                        train_dataset,
                        args.name,
                        train_eval_dir,
                    )
                    best_evaluation = evaluation_loss
                    was_best = True
                # Print
                avg_train_loss /= len(train_dataloader)
                for i, (name, l) in enumerate(avg_train_losses):
                    avg_train_losses[i] = (name, l / len(train_dataloader))
                print(
                    "Epoch: {} // Train Loss: {:.4f} // Time: {:.1f} ({:.1f})".format(
                        epoch,
                        avg_train_loss,
                        time.time() - epoch_time,
                        time.time() - start_time,
                    )
                )
                losses_str = " " * len("Epoch: {}".format(epoch)) + " // "
                for name, l in avg_train_losses:
                    losses_str += "{}: {:.4f} // ".format(name, l)
                print(losses_str)
                print(
                    " " * len("Epoch: {}".format(epoch))
                    + " // Eval Loss: {:.4f} // MPJPE: {:.4f} // MPEEPE: {:.4f}".format(
                        evaluation_loss, mpjpe, mpeepe
                    )
                    + ("*" if was_best else "")
                )

    end_time = time.time()
    print("Training Time:", end_time - start_time)

    # Load Best Model -> Save and/or Evaluate
    load_model(generator_model, generator_path, train_data, device)
    results = evaluate_generator(generator_model, train_data, eval_dataset)

    mpjpe, mpeepe = eval_save_result(results, train_dataset.means, train_dataset.stds, eval_dir, device)
    evaluation_loss = mpjpe + mpeepe

    print("Evaluate Loss: {}".format(evaluation_loss))
    print("Mean Per Joint Position Error: {}".format(mpjpe))
    print("Mean End Effector Position Error: {}".format(mpeepe))


def eval_save_result(results, train_means, train_stds, eval_dir, device, save=True):
    # Save Result
    array_mpjpe = np.empty((len(results),))
    array_mpeepe = np.empty((len(results),))
    for step, (res, res_disp, bvh, filename) in enumerate(results):
        if save:
            eval_path, eval_filename = result_to_bvh(res, res_disp, train_means, train_stds, bvh, filename)
            # Evaluate Positional Error
            mpjpe, mpeepe = eval_metrics.eval_pos_error(
                get_bvh_from_disk(eval_dir, filename),
                get_bvh_from_disk(eval_path, eval_filename),
                device,
                downsample_gt=param["downsample"],
            )
        else:
            result_to_bvh(res, res_disp, train_means, train_stds, bvh, None, save=False)
            # Evaluate Positional Error
            mpjpe, mpeepe = eval_metrics.eval_pos_error(
                get_bvh_from_disk(eval_dir, filename),
                bvh,
                device,
                downsample_gt=param["downsample"],
            )

        array_mpjpe[step] = mpjpe
        array_mpeepe[step] = mpeepe

    return np.mean(array_mpjpe), np.mean(array_mpeepe)


def load_model(model, model_path, train_data, device):
    model_name = os.path.basename(model_path)[: -len(".pt")]
    assert model_name == "generator"
    if model_name == "generator":
        data_path = model_path[: -len("generator.pt")] + "data.pt"
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    data = torch.load(data_path, map_location=device)
    means = data["means"]
    stds = data["stds"]
    train_data.set_means(means["dqs"], means["displacement"])
    train_data.set_stds(stds["dqs"], stds["displacement"])
    return means, stds


def get_model_paths(name, train_eval_dir):
    model_name = "model_" + name + "_" + os.path.basename(os.path.normpath(train_eval_dir))
    model_dir = os.path.join("models", model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    data_path = os.path.join(model_dir, "data.pt")
    generator_path = os.path.join(model_dir, "generator.pt")
    parameters_path = os.path.join(model_dir, "parameters.txt")
    temporal_path = os.path.join(model_dir, "temporal.pt")
    return data_path, generator_path, parameters_path, temporal_path


def save_model(
    generator_model,
    temporal_model,
    train_dataset,
    name,
    train_eval_dir,
):
    data_path, generator_path, parameters_path, temporal_path = get_model_paths(name, train_eval_dir)

    if train_dataset is not None:
        torch.save(
            {
                "means": train_dataset.means,
                "stds": train_dataset.stds,
            },
            data_path,
        )
    if generator_model is not None:
        torch.save(
            {
                "model_state_dict": generator_model.state_dict(),
            },
            generator_path,
        )
        with open(parameters_path, "w") as f:
            f.write(str(param))
    if temporal_model is not None:
        torch.save(
            {
                "model_state_dict": temporal_model[0].state_dict(),
                "means_latent": temporal_model[1],
                "stds_latent": temporal_model[2],
            },
            temporal_path,
        )


def get_bvh_from_disk(path, filename):
    path = os.path.join(path, filename)
    bvh = BVH()
    bvh.load(path)
    return bvh


def get_info_from_bvh(bvh):
    rot_roder = np.tile(bvh.data["rot_order"], (bvh.data["rotations"].shape[0], 1, 1))
    rots = quat.unroll(
        quat.from_euler(np.radians(bvh.data["rotations"]), order=rot_roder),
        axis=0,
    )
    rots = quat.normalize(rots)  # make sure all quaternions are unit quaternions
    pos = bvh.data["positions"]
    parents = bvh.data["parents"]
    parents[0] = 0  # BVH sets root as None
    offsets = bvh.data["offsets"]
    offsets[0] = np.zeros(3)  # force to zero offset for root joint
    return rots, pos, parents, offsets, bvh


def evaluate_generator(generator_model, train_data, dataset):
    # WARNING: means and stds for the model are not set in this function... they should be set before
    generator_model.eval()
    results = []
    with torch.no_grad():
        for index in range(dataset.get_len()):
            norm_motion = dataset.get_item(index)
            # norm_motion has shape (frames, n_joints * 8)
            # norm_motion_dqs = norm_motion["dqs"][
            #     : -(norm_motion["dqs"].shape[0] % param["window_size"]), :
            # ]
            # norm_motion_displacement = norm_motion["displacement"][
            #     : -(norm_motion["displacement"].shape[0] % param["window_size"]), :
            # ]
            # transform to (frames // window_size, window_size, n_joints * 8)
            # norm_motion_dqs = norm_motion_dqs.reshape(
            #     norm_motion_dqs.shape[0] // param["window_size"],
            #     param["window_size"],
            #     -1,
            # )
            # norm_motion_displacement = norm_motion_displacement.reshape(
            #     norm_motion_displacement.shape[0] // param["window_size"],
            #     param["window_size"],
            #     -1,
            # )
            # for SINGLE POSE transform to (frames, 1, n_joints * 8)
            norm_motion_dqs = norm_motion["dqs"].unsqueeze(1)
            norm_motion_displacement = norm_motion["displacement"].unsqueeze(1)
            train_data.set_motions(
                norm_motion["offsets"].unsqueeze(0),
                norm_motion_dqs,
                norm_motion_displacement,
            )
            res_motion, res_displacement = generator_model.forward()
            # res has shape (frames // window_size, n_joints * channels_joint, window_size)
            # transform to (frames, n_joints * channels_joint)
            res_motion = res_motion.permute(0, 2, 1)
            res_motion = res_motion.flatten(0, 1).unsqueeze(0)
            res_motion = res_motion.permute(0, 2, 1)
            res_displacement = res_displacement.permute(0, 2, 1)
            res_displacement = res_displacement.flatten(0, 1).unsqueeze(0)
            res_displacement = res_displacement.permute(0, 2, 1)
            bvh, filename = dataset.get_bvh(index)
            results.append((res_motion, res_displacement, bvh, filename))
    return results


def run_set_data(train_data, dataset):
    with torch.no_grad():
        norm_motion = dataset.get_item()
        train_data.set_motions(
            norm_motion["offsets"].unsqueeze(0),
            norm_motion["dqs"].unsqueeze(0),
            norm_motion["denorm_offsets"].unsqueeze(0),
        )


def run_generator(model):
    # WARNING: means and stds for the model are not set in this function... they should be set before
    model.eval()
    with torch.no_grad():
        res_decoder_motion, res_decoder_displacement = model.forward()
    return res_decoder_motion, res_decoder_displacement


def from_root_quat(q: np.array, parents: np.array) -> np.array:
    """
    Convert root-centered dual quaternion to the skeleton information.

    Parameters
    ----------
    dq: np.array[..., n_joints, 4]
        Includes as first element the global position of the root joint
    parents: np.array[n_joints]

    Returns
    -------
    rotations : np.array[..., n_joints, 4]
    """
    n_joints = q.shape[1]
    # rotations has shape (frames, n_joints, 4)
    rotations = q.copy()
    # make transformations local to the parents
    # (initially local to the root)
    for j in reversed(range(1, n_joints)):
        parent = parents[j]
        if parent == 0:  # already in root space
            continue
        inv = quat.inverse(rotations[..., parent, :])
        rotations[..., j, :] = quat.mul(inv, rotations[..., j, :])
    return rotations


def result_to_bvh(
    res,
    res_disp,
    means,
    stds,
    bvh,
    filename,
    save=True,
    res_global_pos=None,
    are_root_rot_incr=True,
    correct_drift_frames=64,
):
    res = res.permute(0, 2, 1)
    res = res.flatten(0, 1)
    res = res.cpu().detach().numpy()
    if res_disp is not None:
        res_disp = res_disp.permute(0, 2, 1)
        res_disp = res_disp.flatten(0, 1)
        res_disp = res_disp.cpu().detach().numpy()
        disp = res_disp
        disp = disp * stds["displacement"].cpu().numpy() + means["displacement"].cpu().numpy()
    if res_global_pos is not None:
        res_global_pos = res_global_pos.permute(0, 2, 1)
        res_global_pos = res_global_pos.flatten(0, 1)
        res_global_pos = res_global_pos.cpu().detach().numpy()

    # get qs
    qs = res
    # denormalize
    channels_per_joint = 4
    qs = (
        qs * stds["dqs"].reshape((-1, 8))[:, :channels_per_joint].flatten().cpu().numpy()
        + means["dqs"].reshape((-1, 8))[:, :channels_per_joint].flatten().cpu().numpy()
    )
    qs = qs.reshape(qs.shape[0], -1, channels_per_joint)
    if are_root_rot_incr:
        # incremental root rotations to world rotations
        # every correct_drift_frames frames, the root rotation is reset to the bvh
        # to avoid constant drift
        bvh_rots, _, _, _, _, _ = bvh.get_data()
        for i in range(0, qs.shape[0], correct_drift_frames):
            qs[i, 0, :4] = bvh_rots[i, 0, :]
            for j in range(1, correct_drift_frames):
                if i + j >= qs.shape[0]:
                    break
                qs[i + j, 0, :4] = quat.mul(qs[i + j - 1, 0, :4], qs[i + j, 0, :4])
    # get rotations from root space to local space
    rots = from_root_quat(qs, np.array(bvh.data["parents"]))
    # quaternions to euler
    rot_roder = np.tile(bvh.data["rot_order"], (rots.shape[0], 1, 1))
    rotations = np.degrees(quat.to_euler(rots, order=rot_roder))
    bvh.data["rotations"] = rotations
    # positions
    positions = bvh.data["positions"][: rotations.shape[0]]
    if res_global_pos is not None:
        positions[:, 0, :] = res_global_pos
    else:
        # displacement from root local to world space
        world_dis = quat.mul_vec(rots[..., 0, :], disp)
        # correct displacement drift similar to rotations
        for i in range(0, positions.shape[0], correct_drift_frames):
            for j in range(1, correct_drift_frames):
                if i + j >= positions.shape[0]:
                    break
                positions[i + j, 0, :] = positions[i + j - 1, 0, :] + world_dis[i + j, :]
    bvh.data["positions"] = positions
    # save
    path = None
    if save:
        path = "data"
        filename = "eval_" + filename
        bvh.save(os.path.join(path, filename))
    return path, filename


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    parser = argparse.ArgumentParser(description="Train Pose Generator VAE")
    parser.add_argument(
        "data_path",
        type=str,
        help="path to data directory containing one or multiple .bvh for training, last .bvh is used as test data",
    )
    parser.add_argument(
        "name",
        type=str,
        help="name of the experiment, used to save the model and the logs",
    )
    parser.add_argument(
        "--load",
        action="store_true",
        help="load the model(s) from a checkpoint",
    )
    parser.add_argument(
        "--fk",
        action="store_true",
        help="use forward kinematics loss during training",
    )
    args = parser.parse_args()
    main(args)
