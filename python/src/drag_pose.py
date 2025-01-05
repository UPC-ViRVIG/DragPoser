import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import train_temporal
import pymotion.rotations.quat_torch as quat

from utils import fk_rotmat, from_root_quat_to_rotmat


class DragPose:
    def __init__(self, generator_model, temporal_model, means_latent, stds_latent, device, device_gpu):
        self.channels_per_joint = 4
        self.autoencoder = generator_model.autoencoder
        self.encoder = generator_model.autoencoder.encoder
        self.decoder = generator_model.autoencoder.decoder
        self.temporal = temporal_model
        self.data = generator_model.data
        self.parents = generator_model.parents
        self.device = device
        self.device_gpu = device_gpu
        # freeze decoder
        for param in self.decoder.parameters():
            param.requires_grad = False
        self.mse = nn.MSELoss()
        self.means_dqs = (
            self.data.mean_dqs.clone().reshape((-1, 8))[:, : self.channels_per_joint].flatten().unsqueeze(-1)
        )
        self.means_displacement = self.data.mean_displacement.clone().unsqueeze(-1)
        self.stds_dqs = (
            self.data.std_dqs.clone().reshape((-1, 8))[:, : self.channels_per_joint].flatten().unsqueeze(-1)
        )
        self.stds_displacement = self.data.std_displacement.clone().unsqueeze(-1)
        # temporal variables
        param = train_temporal.param
        past_size = param["future_frames"][0]
        self.latent_buffer = torch.empty(
            (past_size, param["latent_dim"]),
            device=temporal_model.device,
        )
        self.temporal_frames_index = param["past_frames"]
        self.means_latent = means_latent
        self.stds_latent = stds_latent
        self.target_latent_buffer = None

    def set_initial_pose(self, initial_pose, init_global_pos, initial_global_rot, initial_heights):
        self.current_global_pos = init_global_pos
        self.current_global_rot = initial_global_rot.squeeze(-1)
        self.latent, _, _ = self.autoencoder.forward_encoder(initial_pose)
        self.latent = self.latent.detach().requires_grad_()
        # self.latent has shape (1, latent_dim)
        self.latent_buffer = torch.tile(self.latent, (self.latent_buffer.shape[0], 1))
        self.latent_buffer = self.latent_buffer.detach()
        # self.latent_buffer has shape (past window size, latent_dim)
        self.displacement_buffer = torch.zeros((self.latent_buffer.shape[0], 3), device=self.device)
        self.displacement_buffer = self.displacement_buffer.detach()
        self.heights_buffer = torch.zeros(
            (self.latent_buffer.shape[0], initial_heights.shape[0]), device=self.device
        )
        self.heights_buffer[:] = initial_heights
        self.heights_buffer = self.heights_buffer.detach()
        # index
        self.current_index = 0

    def loss(
        self,
        input_pose,
        input_displacement,
        target_ee_pos,
        target_ee_rot,
        target_latent,
        offsets,
        mask_joints,
        weights_joints,
        lambda_rot,
        lambda_temporal,
    ):
        # qs and target_dqs are shape (batch_size, n_joints * 8, frames)
        qs = input_pose[..., -1].unsqueeze(-1).clone()
        displacement = input_displacement[..., -1].unsqueeze(-1).clone()

        # denormalize
        qs = qs * self.stds_dqs + self.means_dqs
        displacement = displacement * self.stds_displacement + self.means_displacement

        # change qs and target_qs root rotation from incremental to world space
        world_rotation = quat.mul(
            self.current_global_rot.detach().clone().unsqueeze(0),
            qs.permute(0, 2, 1)[..., :4],
        )
        qs[:, :4, :] = world_rotation.permute(0, 2, 1)

        # change joints from root space to local space
        rotmats_local = from_root_quat_to_rotmat(
            qs.permute(0, 2, 1).reshape((qs.shape[0], qs.shape[-1], -1, self.channels_per_joint)),
            torch.tensor(self.parents, device=self.device),
        )

        # displacement from root local to world space
        displacement = displacement.permute(0, 2, 1)
        world_displacement = quat.mul_vec(world_rotation, displacement)

        # IMPORTANT: FK are executed in world space, but the origin is
        #            the previous pose's root position

        # Forward Kinematics
        pos_qs, rotmats_qs = fk_rotmat(
            rotmats_local,
            world_displacement,
            offsets,
            self.parents,
        )

        # Loss Pos
        loss_pos = (
            ((pos_qs[:, :, mask_joints, :] - target_ee_pos) ** 2)
            * weights_joints[:, 0].unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        ).mean()
        # Loss Rot
        loss_rot = (
            ((rotmats_qs[:, :, mask_joints, :] - target_ee_rot) ** 2)
            * weights_joints[:, 1].unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        ).mean()

        # Loss Temporal
        loss_temporal = self.mse(self.latent.squeeze(0), target_latent)

        # Additional Losses ----------------------------------------------
        # global_pos = self.current_global_pos.detach().clone()[0, :, 0]
        # y_axis = 1  # Xsens -> 1

        # # Loss Feet Floor
        # floor_level = 0.0  # Xsens -> 0.0
        # loss_feet_floor = ((global_pos[y_axis] + (pos_qs[:, :, [4, 8], y_axis] - floor_level)) ** 2).mean()

        # # Loss Head Hips Forward
        # fwd = torch.tensor([0, 0, 1], device=self.device)
        # fwd_head = quat.mul_vec(quat.from_matrix(rotmats_qs[0, 0, 13, :, :]), fwd)
        # fwd_head[1] = 0
        # fwd_head_norm = torch.norm(fwd_head)
        # if fwd_head_norm > 0.5:
        #     fwd_head = fwd_head / fwd_head_norm
        #     fwd_hips = quat.mul_vec(quat.from_matrix(rotmats_qs[0, 0, 0, :, :]), fwd)
        #     fwd_hips[1] = 0
        #     fwd_hips = fwd_hips / torch.norm(fwd_hips)
        #     loss_head_hips_forward = (
        #         1
        #         - torch.min(
        #             torch.tensor(1, device=self.device),
        #             torch.sum(fwd_head * fwd_hips, dim=-1) + 0.2,
        #         )
        #     ) ** 2
        # else:
        #     loss_head_hips_forward = torch.tensor(0, device=self.device)

        # # Loss Head Hips Colinear
        # pos_head = pos_qs[0, 0, 13] + global_pos
        # pos_head[y_axis] = 0  # project to the ground
        # pos_hips = pos_qs[0, 0, 0] + global_pos
        # pos_hips[y_axis] = 0
        # loss_head_hips_colinear = torch.sum((pos_head - pos_hips) ** 2)

        # # Loss Hips Feet Colinear
        # pos_left_foot = pos_qs[0, 0, 3] + global_pos
        # pos_left_foot[y_axis] = 0
        # pos_right_foot = pos_qs[0, 0, 7] + global_pos
        # pos_right_foot[y_axis] = 0
        # loss_hips_feet_colinear = torch.max(
        #     torch.sum((pos_hips - pos_left_foot) ** 2) - 0.2 * 0.2,
        #     torch.tensor(0, device=self.device),
        # ) + torch.max(
        #     torch.sum((pos_hips - pos_right_foot) ** 2) - 0.2 * 0.2,
        #     torch.tensor(0, device=self.device),
        # )

        # additional_losses = (
        #     loss_head_hips_forward
        #     + loss_head_hips_colinear
        #     + loss_feet_floor
        #     + loss_hips_feet_colinear
        # )
        additional_losses = 0

        return (
            loss_pos,
            loss_rot * lambda_rot,
            loss_temporal * lambda_temporal,
            additional_losses,
            world_displacement,
            displacement,
            world_rotation,
            pos_qs,
        )

    def run(
        self,
        target_ee_pos,
        target_ee_rot,
        mask_joints,
        weights_joints,
        offsets,
        stop_eps_pos=1e-2,
        stop_eps_rot=1e-2,
        max_iter=100,
        min_loss_incr=0.00001,
        learning_rate=1e-3,
        lambda_rot=1,
        lambda_temporal=1,
        temporal_future_window=60,
        height_indices=[0, 4, 8, 13, 17, 21],
        joint_adjustment_indices=None,  # None or (joint_index, end_effector_index)
        joint_adjustment_weight=0.01,
        verbose=False,
    ):
        init_max_iter = max_iter

        optimizer = optim.Adam([self.latent], lr=learning_rate)

        # Init some variables
        loss = float("inf")
        loss_pos = float("inf")
        loss_temporal = float("inf")
        decoder_time = 0
        loss_time = 0
        backward_time = 0
        # positions are in global space but with the root of the character at the origin
        # target_ee_pos has shape (end_effectors, 3)
        target_ee_pos = target_ee_pos.unsqueeze(0).unsqueeze(0)
        # rotations are in global space
        # target_ee_rot has shape (end_effectors, 3, 3)
        target_ee_rot = target_ee_rot.unsqueeze(0).unsqueeze(0)

        # Temporal -------------------------------
        sample_step = train_temporal.param["sample_step"]
        assert temporal_future_window % sample_step == 0
        if (
            self.target_latent_buffer is None
            or self.target_latent_buffer.shape[0] != temporal_future_window + 1
        ):
            # self.target_latent_buffer has shape (temporal_future_window + 1, latent_dim)
            self.target_latent_buffer = torch.zeros(
                (temporal_future_window + 1, self.latent.shape[-1]), device=self.temporal.device
            )
        start_temporal_time = time.time()
        if self.current_index == 0:
            with torch.no_grad():
                self.temporal.eval()
                input_latent = self.latent_buffer[self.temporal_frames_index][:-1].unsqueeze(0).clone()
                input_displacement = (
                    self.displacement_buffer[self.temporal_frames_index][:-1].unsqueeze(0).clone()
                )
                # compute accumulated displacements
                for i, j in enumerate(self.temporal_frames_index[:-1]):
                    input_displacement[:, i] = torch.sum(self.displacement_buffer[j : j + sample_step], dim=0)
                # get input target latent
                input_target_latent = (
                    self.latent_buffer[self.temporal_frames_index][-1].unsqueeze(0).unsqueeze(0)
                ).clone()
                # normalize
                input_latent = (input_latent - self.means_latent) / self.stds_latent
                input_target_latent = (input_target_latent - self.means_latent) / self.stds_latent
                # root height
                input_heights = self.heights_buffer[self.temporal_frames_index][:-1].unsqueeze(0).clone()
                # concatenate latent and displacement
                input_latent = torch.cat((input_latent, input_displacement, input_heights), dim=-1)
                # Forward Temporal
                input_latent = input_latent.to(self.device_gpu)
                input_target_latent = input_target_latent.to(self.device_gpu)
                self.target_latent_buffer = self.target_latent_buffer.to(self.device_gpu)
                # self.target_latent_buffer has shape (temporal_future_window + 1, latent_dim)
                for i in range(0, temporal_future_window + 1, sample_step):
                    target_latent = self.temporal(input_latent, input_target_latent)
                    # target_latent has shape (latent_dim,)
                    input_target_latent = torch.cat((input_target_latent, target_latent[0:1, -1:]), dim=1)
                    # update target_latent_buffer
                    self.target_latent_buffer[i] = target_latent[0, -1]
                self.target_latent_buffer = self.target_latent_buffer.to(self.device)
                # denormalize
                self.target_latent_buffer = self.target_latent_buffer * self.stds_latent + self.means_latent
                # linearly interpolate to upsample
                for i in list(range(0, temporal_future_window, sample_step)):
                    # NOTE: I tried to linearly interpolate the whole buffer at once,
                    #       but it works best if the target is the immediate next prediction
                    self.target_latent_buffer[i : i + sample_step + 1] = torch.lerp(
                        self.target_latent_buffer[i].unsqueeze(0),
                        self.target_latent_buffer[i + sample_step].unsqueeze(0),
                        torch.linspace(1, 1, sample_step + 1, device=self.device).unsqueeze(-1),
                    )  # to linearly interpolate change to torch.linspace(0, 1, ...)
                self.target_latent_buffer = self.target_latent_buffer.detach()
        temporal_time = time.time() - start_temporal_time

        # get this frame target_latent:
        target_latent = self.target_latent_buffer[self.current_index]

        # Optimize -------------------------------
        prev_loss = 10000000
        loss_incr = 1
        iter = 0
        while (
            (loss_pos > stop_eps_pos or loss_rot > stop_eps_rot)
            and max_iter > 0
            and loss_incr > min_loss_incr
        ):
            # Decoder
            start_decoder_time = time.time()
            # output_pose has shape [batch_size, num_joints * channels_per_joint, frames]
            # output_displacement has shape [batch_size, 3, frames]
            self.current_latent = self.latent.clone().detach()
            output_pose, output_displacement = self.decoder(
                self.latent, self.data.mean_dqs, self.data.std_dqs
            )
            decoder_time += time.time() - start_decoder_time

            # Loss
            start_loss_time = time.time()
            (
                loss_pos,
                loss_rot,
                loss_temporal,
                additional_losses,
                world_displacement,
                displacement,  # root space
                world_rotation,
                world_pos_wrt_prev_frame,
            ) = self.loss(
                output_pose,
                output_displacement,
                target_ee_pos,
                target_ee_rot,
                target_latent,
                offsets,
                mask_joints,
                weights_joints,
                lambda_rot,
                lambda_temporal,
            )
            loss = loss_pos + loss_rot + loss_temporal + additional_losses
            loss_time += time.time() - start_loss_time
            # gradient of loss w.r.t. latent
            start_backward_time = time.time()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = loss.item()
            loss_pos = loss_pos.item()
            loss_rot = loss_rot.item()
            loss_temporal = loss_temporal.item()
            backward_time += time.time() - start_backward_time
            # Increment
            max_iter -= 1
            iter += 1
            # Update aux variables
            loss_incr = prev_loss - loss
            prev_loss = loss

        decoder_time /= iter
        loss_time /= iter
        backward_time /= iter

        if verbose:
            print(
                f"Loss sqrt(Pos): {np.sqrt(loss_pos):.5f} // Loss Rot: {loss_rot:.5f} // Loss Temporal: {loss_temporal:.5f} // Iter: {init_max_iter - max_iter}"
            )
            print(
                f"Mean decoder time: {decoder_time:.5f} // Temporal time: {temporal_time:.5f} // Mean Loss Time: {loss_time:.5f} // Mean backward time: {backward_time:.5f}"
            )

        # Update global position and rotation
        self.current_global_pos = self.current_global_pos + world_displacement.permute(0, 2, 1)
        self.current_global_rot = world_rotation.squeeze(0)

        # Joint adjustment
        if joint_adjustment_indices is not None:
            joint_index = joint_adjustment_indices[0]
            ee_index = joint_adjustment_indices[1]
            ee_pos = target_ee_pos[0, 0, ee_index]
            joint_pos = world_pos_wrt_prev_frame[0, 0, joint_index]
            joint_adjustment = (ee_pos - joint_pos) * joint_adjustment_weight
            self.current_global_pos[0, :, 0] += joint_adjustment
            displacement += joint_adjustment.unsqueeze(0).unsqueeze(0)

        # Update buffers
        self.latent_buffer[:-1] = self.latent_buffer[1:].clone()
        self.latent_buffer[-1] = self.current_latent
        self.displacement_buffer[:-1] = self.displacement_buffer[1:].clone()
        self.displacement_buffer[-1] = displacement.squeeze(0)
        self.heights_buffer[:-1] = self.heights_buffer[1:].clone()
        self.heights_buffer[-1] = (world_pos_wrt_prev_frame + self.current_global_pos[0, :, 0])[
            0, 0, height_indices, 1
        ]

        # Update output pose
        output_pose[..., :4, -1] = (
            self.current_global_rot - self.means_dqs.squeeze(-1)[:4]
        ) / self.stds_dqs.squeeze(-1)[:4]

        # Update others
        if temporal_future_window == 0:
            self.current_index = 0
        else:
            self.current_index = (self.current_index + 1) % temporal_future_window

        # DEBUG
        # temporal_output_pose, temporal_output_displacement = self.decoder(
        #     target_latent.unsqueeze(0),
        #     self.data.mean_dqs,
        #     self.data.std_dqs,
        # )
        # temporal_output_pose[..., :4, -1] = (
        #     self.current_global_rot - self.means_dqs.squeeze(-1)[:4]
        # ) / self.stds_dqs.squeeze(-1)[:4]

        return output_pose[..., -1][0], self.current_global_pos[..., -1][0]
