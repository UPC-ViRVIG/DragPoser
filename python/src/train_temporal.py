import train
import argparse
import random
import torch
import os
import time
import warnings
import numpy as np
from torch.utils.data.dataloader import DataLoader
from motion_data import TestMotionData, TrainMotionData
from train_data import Train_Data
from generator_architecture import Generator_Model
from temporal_transformer import Temporal

sample_step = 4

param = {
    "batch_size": 512,
    "epochs": 80,
    "learning_rate": 1e-3,
    "window_size": 120,
    "past_frames": [i for i in range(0, 60, sample_step)],
    "future_frames": [i for i in range(60, 120, sample_step)],
    "window_step": 16,
    "downsample": 1,
    "features_transformer": train.param["latent_dim"] * 2,
    "n_heads": 4,
    "n_encoder_layers": 3,
    "n_decoder_layers": 3,
    "dim_feedforward": 2048,
    "dropout": 0.1,
    "latent_dim": train.param["latent_dim"],
    "lambda_displacement": 10.0,
    "sample_step": sample_step,
    "height_indices": [0, 4, 8, 13, 17, 21],
    "limbs_random_prob": 0.1,
}

left_arm_indices = [14, 15, 16, 17]
right_arm_indices = [18, 19, 20, 21]
left_leg_indices = [1, 2, 3, 4]
right_leg_indices = [5, 6, 7, 8]


def main(args):
    # Set seed
    torch.manual_seed(train.param["seed"])
    random.seed(train.param["seed"])
    np.random.seed(train.param["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    # Additional Info when using cuda
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))

    # Data Paths
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

    # Get reference_parents
    reference_parents = None  # used to make sure all bvh have the same structure
    for filename in train_files:
        if filename[-4:] == ".bvh":
            _, _, parents, _, _ = train.get_info_from_bvh(train.get_bvh_from_disk(train_dir, filename))
            if reference_parents is None:
                reference_parents = parents.copy()
                break

    # Create Models
    train_data = Train_Data(device, param, None)
    generator_model = Generator_Model(device, train.param, reference_parents, train_data).to(device)
    temporal_model = Temporal(param, device).to(device)

    # Load model
    _, generator_path, _, temporal_path = train.get_model_paths(args.name, train_eval_dir)
    means, stds = train.load_model(generator_model, generator_path, train_data, device)

    # Prepare Data
    train_dataset = TrainMotionData(param, train.scale, train_eval_dir, device)
    eval_dataset = TestMotionData(param, train.scale, device)

    # Train Files
    for filename in train_files:
        if filename[-4:] == ".bvh":
            rots, pos, parents, offsets, _ = train.get_info_from_bvh(
                train.get_bvh_from_disk(train_dir, filename)
            )
            assert np.all(reference_parents == parents)  # make sure all bvh have the same structure
            # Train Dataset
            train_dataset.add_motion(
                offsets,
                pos[:, 0, :],  # only global position
                rots,
                parents,
                temporal=True,
            )
    # Once all train files are added, compute the means and stds and normalize
    train_dataset.normalize_force(means, stds)
    eval_dataset.set_means_stds(means, stds)
    # Eval Files
    for filename in eval_files:
        if filename[-4:] == ".bvh":
            rots, pos, parents, offsets, bvh = train.get_info_from_bvh(
                train.get_bvh_from_disk(eval_dir, filename)
            )
            assert np.all(reference_parents == parents)  # make sure all bvh have the same structure
            # Eval Dataset
            eval_dataset.add_motion(
                offsets,
                pos[:, 0, :],  # only global position
                rots,
                parents,
                bvh,
                filename,
                temporal=True,
            )
    # Once all eval files are added, normalize
    eval_dataset.normalize()

    # Dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=param["batch_size"], shuffle=True)

    # Load Temporal
    best_evaluation = float("inf")
    if args.load:
        means_latent, stds_latent = load_model(temporal_model, temporal_path, device)
        # Check previous best evaluation loss
        eval_loss = evaluate(
            eval_dataset, generator_model, temporal_model, train_data, means_latent, stds_latent
        )
        best_evaluation = eval_loss
    else:
        # Compute mean and std of the latent vectors
        latent_buffer = None
        with torch.no_grad():
            for i in range(0, len(train_dataset)):
                (motion, norm_motion), (_, _) = train_dataset[i]
                train_data.set_motions(
                    motion["offsets"],
                    torch.cat((norm_motion["dqs_past"], norm_motion["dqs_future"]), dim=0)
                    .unsqueeze(0)
                    .flatten(start_dim=0, end_dim=1)
                    .unsqueeze(1),
                    torch.cat(
                        (
                            norm_motion["displacement_past"],
                            norm_motion["displacement_future"],
                        ),
                        dim=0,
                    )
                    .unsqueeze(0)
                    .flatten(start_dim=0, end_dim=1)
                    .unsqueeze(1),
                    None,
                    None,
                )
                generator_model.eval()
                latent, _, _ = generator_model.autoencoder.forward_encoder(train_data.motion)
                if latent_buffer is None:
                    latent_buffer = latent
                else:
                    latent_buffer = torch.cat((latent_buffer, latent), dim=0)
        means_latent = torch.mean(latent_buffer, dim=0)
        stds_latent = torch.std(latent_buffer, dim=0)

    # Training Loop
    best_evaluation = float("inf")
    start_time = time.time()
    for epoch in range(param["epochs"]):
        avg_train_loss = 0.0
        count = 0
        epoch_time = time.time()
        for step, (
            (motion, norm_motion),
            (_, next_norm_motion),
        ) in enumerate(train_dataloader):
            if norm_motion["dqs_past"].shape[0] != param["batch_size"]:
                continue

            # Forward Encoders -- Past
            dqs_past = norm_motion["dqs_past"].flatten(start_dim=0, end_dim=1).unsqueeze(1)

            def apply_limbs_mask(indices):
                dqs_past.reshape((param["batch_size"], dqs_past.shape[0] // param["batch_size"], 1, -1, 8))[
                    :, :-1, :, indices, :
                ] = (
                    torch.randn(
                        (
                            param["batch_size"],
                            dqs_past.shape[0] // param["batch_size"] - 1,
                            1,
                            len(indices),
                            8,
                        ),
                        device=device,
                    )
                    * train_data.std_dqs.reshape(-1, 8)[indices]
                    + train_data.mean_dqs.reshape(-1, 8)[indices]
                )

            if random.random() < param["limbs_random_prob"]:
                apply_limbs_mask(left_arm_indices)
            if random.random() < param["limbs_random_prob"]:
                apply_limbs_mask(right_arm_indices)
            if random.random() < param["limbs_random_prob"]:
                apply_limbs_mask(left_leg_indices)
            if random.random() < param["limbs_random_prob"]:
                apply_limbs_mask(right_leg_indices)

            train_data.set_motions(
                motion["offsets"],
                dqs_past,
                norm_motion["displacement_past"].flatten(start_dim=0, end_dim=1).unsqueeze(1),
                None,
                None,
            )
            generator_model.eval()
            with torch.no_grad():
                # use mu + logvar to sample latent
                # as "data augmentation"
                latent, _, _ = generator_model.autoencoder.forward_encoder(train_data.motion)
            latent = latent.reshape(param["batch_size"], -1, train.param["latent_dim"]).detach()
            # Forward Encoders -- Future
            train_data.set_motions(
                motion["offsets"],
                norm_motion["dqs_future"].flatten(start_dim=0, end_dim=1).unsqueeze(1),
                norm_motion["displacement_future"].flatten(start_dim=0, end_dim=1).unsqueeze(1),
                None,
                None,
            )
            generator_model.eval()
            with torch.no_grad():
                latent_target, _, _ = generator_model.autoencoder.forward_encoder(train_data.motion)
            latent_target = latent_target.reshape(param["batch_size"], -1, train.param["latent_dim"]).detach()
            # Forward Temporal Model
            temporal_model.train()
            # normalize
            latent = (latent - means_latent) / stds_latent
            latent_target = (latent_target - means_latent) / stds_latent
            # add accumulated displacements to latent
            disp_past_acc = motion["displacement_past_acc"]
            heights = motion["heights"]
            latent = torch.cat((latent, disp_past_acc, heights), dim=-1)
            # we will shift everything one to the right so that
            # the last past predicts the first target,
            # the second target predicts the third target, and so on
            latent_input = latent[:, :-1, :]
            additional_input_dim = 3 + len(
                param["height_indices"]
            )  # remove accumulated displacements + heights
            latent_input_target = torch.cat(
                (
                    latent[:, latent.shape[1] - 1 : latent.shape[1], :-additional_input_dim],
                    latent_target[:, :-1, :],
                ),
                dim=1,
            )
            # get target mask
            tgt_mask = temporal_model.get_tgt_mask(latent_input_target.shape[1]).to(device)
            # forward temporal
            temporal_model.forward(latent_input, latent_input_target, tgt_mask=tgt_mask)
            # Loss
            loss = temporal_model.optimize_parameters(latent_target)
            avg_train_loss += loss
            count += 1
        # Evaluate
        eval_loss = evaluate(
            eval_dataset,
            generator_model,
            temporal_model,
            train_data,
            means_latent,
            stds_latent,
        )
        was_best = False
        if eval_loss < best_evaluation:
            train.save_model(
                None,
                (temporal_model, means_latent, stds_latent),
                None,
                args.name,
                train_eval_dir,
            )
            best_evaluation = eval_loss
            was_best = True
        # Print
        avg_train_loss /= count
        print(
            "Epoch: {} // Train Loss: {:.4f} // Eval Loss: {:.4f} // Time: {:.1f} ({:.1f})".format(
                epoch,
                avg_train_loss,
                eval_loss,
                time.time() - epoch_time,
                time.time() - start_time,
            )
            + ("*" if was_best else "")
        )

    end_time = time.time()
    print("Training Time:", end_time - start_time)

    # Load best model and evaluate
    means_latent, stds_latent = load_model(temporal_model, temporal_path, device)
    eval_loss = evaluate(
        eval_dataset,
        generator_model,
        temporal_model,
        train_data,
        means_latent,
        stds_latent,
    )
    print("Evaluate Loss: {} ".format(eval_loss))


def evaluate(eval_dataset, generator_model, temporal_model, train_data, means_latent, stds_latent):
    generator_model.eval()
    temporal_model.eval()
    avg_loss = 0.0
    count = 0
    with torch.no_grad():
        for index in range(eval_dataset.get_len()):
            norm_motion = eval_dataset.get_item(index)
            norm_motion_dqs = norm_motion["dqs"].unsqueeze(1)
            norm_motion_displacement = norm_motion["displacement"].unsqueeze(1)
            norm_motion_displacement_acc = norm_motion["displacement_acc"].unsqueeze(1)
            norm_motion_heights = norm_motion["heights"].unsqueeze(1)
            # remove last sample_step frames for the accumulated displacements
            norm_motion_dqs = norm_motion_dqs[:-sample_step, :, :]
            norm_motion_displacement = norm_motion_displacement[:-sample_step, :, :]
            norm_motion_displacement_acc = norm_motion_displacement_acc[:-sample_step, :, :]
            norm_motion_heights = norm_motion_heights[:-sample_step, :, :]
            # norm_motion has shape (frames, 1, n_joints * channels_per_joint)
            norm_motion_dqs = norm_motion_dqs[: -(norm_motion_dqs.shape[0] % param["window_size"])]
            norm_motion_displacement = norm_motion_displacement[
                : -(norm_motion_displacement.shape[0] % param["window_size"])
            ]
            norm_motion_displacement_acc = norm_motion_displacement_acc[
                : -(norm_motion_displacement_acc.shape[0] % param["window_size"])
            ]
            norm_motion_heights = norm_motion_heights[
                : -(norm_motion_heights.shape[0] % param["window_size"])
            ]
            if norm_motion_dqs.shape[0] <= 0:
                continue
            # we want to change to shape (frames // window_size, window_size, n_joints * channels_per_joint)
            norm_motion_dqs = norm_motion_dqs.reshape(
                (
                    norm_motion_dqs.shape[0] // param["window_size"],
                    param["window_size"],
                    -1,
                )
            )
            norm_motion_displacement = norm_motion_displacement.reshape(
                (
                    norm_motion_displacement.shape[0] // param["window_size"],
                    param["window_size"],
                    -1,
                )
            )
            norm_motion_displacement_acc = norm_motion_displacement_acc.reshape(
                (
                    norm_motion_displacement_acc.shape[0] // param["window_size"],
                    param["window_size"],
                    -1,
                )
            )
            norm_motion_heights = norm_motion_heights.reshape(
                (
                    norm_motion_heights.shape[0] // param["window_size"],
                    param["window_size"],
                    -1,
                )
            )
            # selected frames only
            norm_motion_dqs_past = (
                norm_motion_dqs[:, param["past_frames"], :].flatten(start_dim=0, end_dim=1).unsqueeze(1)
            )
            norm_motion_displacement_past = (
                norm_motion_displacement[:, param["past_frames"], :]
                .flatten(start_dim=0, end_dim=1)
                .unsqueeze(1)
            )
            norm_motion_displacement_acc_past = (
                norm_motion_displacement_acc[:, param["past_frames"], :]
                .flatten(start_dim=0, end_dim=1)
                .unsqueeze(1)
            )
            norm_motion_heights_past = (
                norm_motion_heights[:, param["past_frames"], :].flatten(start_dim=0, end_dim=1).unsqueeze(1)
            )
            norm_motion_dqs_future = (
                norm_motion_dqs[:, param["future_frames"], :].flatten(start_dim=0, end_dim=1).unsqueeze(1)
            )
            norm_motion_displacement_future = (
                norm_motion_displacement[:, param["future_frames"], :]
                .flatten(start_dim=0, end_dim=1)
                .unsqueeze(1)
            )
            # forward encoder -- past
            train_data.set_motions(
                norm_motion["offsets"].unsqueeze(0),
                norm_motion_dqs_past,
                norm_motion_displacement_past,
            )
            latent, _, _ = generator_model.autoencoder.forward_encoder(train_data.motion)
            latent = latent.reshape(
                -1,
                len(param["past_frames"]),
                train.param["latent_dim"],
            ).detach()
            # forward encoder -- future
            train_data.set_motions(
                norm_motion["offsets"].unsqueeze(0),
                norm_motion_dqs_future,
                norm_motion_displacement_future,
            )
            latent_target, _, _ = generator_model.autoencoder.forward_encoder(train_data.motion)
            latent_target = latent_target.reshape(
                -1,
                len(param["future_frames"]),
                train.param["latent_dim"],
            ).detach()
            # normalize
            latent = (latent - means_latent) / stds_latent
            latent_target = (latent_target - means_latent) / stds_latent
            # add accumulated displacements to latent
            norm_motion_displacement_acc_past = norm_motion_displacement_acc_past.reshape(
                -1,
                len(param["past_frames"]),
                3,
            )
            # add root height to latent
            norm_motion_heights_past = norm_motion_heights_past.reshape(
                -1,
                len(param["past_frames"]),
                len(param["height_indices"]),
            )
            latent = torch.cat((latent, norm_motion_displacement_acc_past, norm_motion_heights_past), dim=-1)
            # we will shift everything one to the right so that
            # the last past predicts the first target,
            # the second target predicts the third target, and so on
            latent_input = latent[:, :-1, :]
            additional_input_dim = 3 + len(
                param["height_indices"]
            )  # remove accumulated displacements + heights
            latent_input_target = torch.cat(
                (
                    latent[:, latent.shape[1] - 1 : latent.shape[1], :-additional_input_dim],
                    latent_target[:, :-1, :],
                ),
                dim=1,
            )
            # get target mask
            tgt_mask = temporal_model.get_tgt_mask(latent_input_target.shape[1]).to(temporal_model.device)
            # forward temporal
            temporal_model.forward(latent_input, latent_input_target, tgt_mask=tgt_mask)
            # loss
            loss = temporal_model.compute_loss(latent_target)
            loss = loss.item()
            avg_loss += loss
            count += 1
    return avg_loss / count


def load_model(model, model_path, device):
    model_name = os.path.basename(model_path)[: -len(".pt")]
    assert model_name == "temporal"
    if model_name == "temporal":
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        means_latent = checkpoint["means_latent"]
        stds_latent = checkpoint["stds_latent"]
        return means_latent, stds_latent


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    parser = argparse.ArgumentParser(description="Train Temporal Network")
    parser.add_argument(
        "data_path",
        type=str,
        help="path to data directory containing one or multiple .bvh for training, last .bvh is used as test data",
    )
    parser.add_argument(
        "name",
        type=str,
        help="name of the experiment, used to load/save the model and the logs",
    )
    parser.add_argument(
        "--load",
        action="store_true",
        help="load the model(s) from a checkpoint",
    )
    args = parser.parse_args()
    main(args)
