import torch
import torch.nn as nn
import pymotion.rotations.quat_torch as quat
from skeleton import (
    SkeletonPool,
    SkeletonUnpool,
    find_neighbor,
    SkeletonConv,
    create_pooling_list,
)


class Autoencoder(nn.Module):
    def __init__(self, param, parents, device):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(param, parents, device)
        self.decoder = Decoder(param, parents, device)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_encoder(self, input_motion):
        mu, logvar = self.encoder(input_motion)
        latent = self.reparameterize(mu, logvar)
        return latent, mu, logvar

    def forward(self, input_motion, mean_dqs, std_dqs):
        is_consecutive = input_motion.ndim > 3
        if is_consecutive:
            # input_motion has shape (batch_size, 2, num_joints * channels_per_joint, frames)
            # merge batch_size and 2 dimensions
            input_motion = input_motion.reshape(
                (input_motion.shape[0] * input_motion.shape[1],) + input_motion.shape[2:]
            )

        # Forward pass
        latent, mu, logvar = self.forward_encoder(input_motion)
        output_motion, output_displacement = self.decoder(latent, mean_dqs, std_dqs)

        if is_consecutive:
            # reshape output_motion to (batch_size, 2, num_joints * channels_per_joint, frames)
            output_motion = output_motion.reshape((input_motion.shape[0] // 2, 2) + output_motion.shape[1:])
            output_displacement = output_displacement.reshape(
                (input_motion.shape[0] // 2, 2) + output_displacement.shape[1:]
            )
            # reshape mu and logvar to (batch_size, 2, latent_dim)
            mu = mu.reshape((input_motion.shape[0] // 2, 2) + mu.shape[1:])
            logvar = logvar.reshape((input_motion.shape[0] // 2, 2) + logvar.shape[1:])

        # mu and logvar are shape (batch_size, latent_dim) or (batch_size, 2, latent_dim)
        return output_motion, output_displacement, mu, logvar, latent


class Encoder(nn.Module):
    def __init__(self, param, parents, device):
        super(Encoder, self).__init__()

        self.layers = nn.ModuleList()
        self.parents = [parents]
        self.pooling_lists = []

        kernel_size = param["kernel_size_temporal_dim"]
        padding = (kernel_size - 1) // 2
        stride = param["stride_encoder_conv"]

        # Compute pooled skeletons
        number_layers = 3
        layer_parents = parents
        for l in range(number_layers):
            pooling_list, layer_parents = create_pooling_list(layer_parents, add_displacement=False)
            self.pooling_lists.append(pooling_list)
            self.parents.append(layer_parents)

        default_channels_per_joints = 8
        factor = param["channel_factor"]
        self.channel_size = [default_channels_per_joints] + [
            default_channels_per_joints * (factor**i) for i in range(1, number_layers + 1)
        ]  # first each joint has 8 (dual quaternion) channels, then we multiply by 2 every layer to increase the number of conv. filters

        for l in range(number_layers):
            seq = []

            neighbor_list, _ = find_neighbor(
                self.parents[l],
                param["neighbor_distance"],
                add_displacement=False,
            )
            num_joints = len(neighbor_list)

            seq.append(
                SkeletonConv(
                    param=param,
                    neighbor_list=neighbor_list,
                    kernel_size=kernel_size,
                    in_channels_per_joint=self.channel_size[l],
                    out_channels_per_joint=self.channel_size[l + 1],
                    joint_num=num_joints,
                    padding=padding,
                    stride=stride,
                    device=device,
                )
            )

            seq.append(
                SkeletonPool(
                    pooling_list=self.pooling_lists[l],
                    old_parents=self.parents[l],
                    channels_per_edge=self.channel_size[l + 1],
                    device=device,
                )
            )

            seq.append(nn.LeakyReLU(negative_slope=0.2))
            # Append to the list of layers
            self.layers.append(nn.Sequential(*seq))

        # Projection to mu and logvar
        primal_skel_features = self.channel_size[-1] * (len(self.parents[-1]))
        # primal_skel_frames = param["window_size"] // 2**number_layers
        self.f_mu = nn.Linear(
            # primal_skel_features * primal_skel_frames,
            primal_skel_features,
            param["latent_dim"],
        )
        self.f_logvar = nn.Linear(
            # primal_skel_features * primal_skel_frames,
            primal_skel_features,
            param["latent_dim"],
        )
        # init self.f_logvar weights to 0 so that the variance is 1 (e^logvar)
        with torch.no_grad():
            self.f_logvar.weight.copy_(torch.zeros_like(self.f_logvar.weight).to(device))

    def forward(self, input):
        # forward
        for layer in self.layers:
            input = layer(input)
        # mu and logvar will be shape (batch_size, latent_dim)
        mu = self.f_mu(input.flatten(start_dim=1))
        logvar = self.f_logvar(input.flatten(start_dim=1))
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, param, parents, device):
        super(Decoder, self).__init__()

        self.param = param
        self.device = device
        self.layers = nn.ModuleList()
        self.parents = [parents]
        self.pooling_lists = []

        kernel_size = param["kernel_size_temporal_dim"]
        padding = (kernel_size - 1) // 2

        # Compute pooled skeletons
        number_layers = 3
        layer_parents = parents
        for l in range(number_layers):
            pooling_list, layer_parents = create_pooling_list(
                layer_parents, add_displacement=l != number_layers - 1
            )
            self.pooling_lists.append(pooling_list)
            self.parents.append(layer_parents)

        number_layers = 3
        default_channels_per_joints = 4
        factor = param["channel_factor"]
        self.channel_size = [default_channels_per_joints] + [
            default_channels_per_joints * (factor**i) for i in range(1, number_layers + 1)
        ]  # first each joint has default_channels_per_joints channels, after collapse default_channels_per_joints**2, then default_channels_per_joints**4...

        # latent to primal skeleton
        self.primal_skel_features = self.channel_size[-1] * len(self.parents[-1])
        # self.primal_skel_frames = param["window_size"] // 2**number_layers
        self.f_latent = nn.Linear(
            param["latent_dim"],
            # self.primal_skel_features * self.primal_skel_frames,
            self.primal_skel_features,
        )

        for l in range(number_layers):
            seq = []

            neighbor_list, _ = find_neighbor(self.parents[number_layers - l - 1], param["neighbor_distance"])
            num_joints = len(neighbor_list)

            # seq.append(
            #     nn.Upsample(
            #         scale_factor=self.param["upsample_decoder_scale"],
            #         mode="linear",
            #         align_corners=False,
            #     )
            # )
            seq.append(
                SkeletonUnpool(
                    pooling_list=self.pooling_lists[number_layers - l - 1],
                    channels_per_edge=self.channel_size[number_layers - l],
                    device=device,
                )
            )

            seq.append(
                SkeletonConv(
                    param=param,
                    neighbor_list=neighbor_list,
                    kernel_size=kernel_size,
                    in_channels_per_joint=self.channel_size[number_layers - l],
                    out_channels_per_joint=self.channel_size[number_layers - l] // factor,
                    joint_num=num_joints,
                    padding=padding,
                    stride=1,
                    device=device,
                )
            )
            if l != number_layers - 1:
                seq.append(nn.LeakyReLU(negative_slope=0.2))
            # Append to the list of layers
            self.layers.append(nn.Sequential(*seq))

    def forward(self, input, mean_dqs, std_dqs):
        channels_per_joint = 4
        # input has shape (batch_size, latent_dim)
        # Project to primal skeleton
        input = self.f_latent(input).reshape(
            # (input.shape[0], self.primal_skel_features, self.primal_skel_frames)
            (input.shape[0], self.primal_skel_features, 1)
        )
        # Decoder
        for layer in self.layers:
            input = layer(input)
        # input has shape (batch_size, num_joints * channels_per_joint, frames)
        # separate motion from displacement
        motion = input[:, :-channels_per_joint, :]
        displacement = input[:, -channels_per_joint:, :]
        # remove padding from displacement
        displacement = displacement[:, :3, :]
        # denormalize rotations
        motion = motion * std_dqs.reshape((-1, 8))[:, :channels_per_joint].flatten().unsqueeze(
            -1
        ) + mean_dqs.reshape((-1, 8))[:, :channels_per_joint].flatten().unsqueeze(-1)
        # change input to shape (batch_size, frames, num_joints, channels_per_joint)
        motion = motion.reshape(motion.shape[0], -1, channels_per_joint, motion.shape[-1]).permute(0, 3, 1, 2)
        # convert to unit dual quaternions
        motion = quat.normalize(motion)
        # normalize rotations
        motion = motion.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)
        motion = (
            motion - mean_dqs.reshape((-1, 8))[:, :channels_per_joint].flatten().unsqueeze(-1)
        ) / std_dqs.reshape((-1, 8))[:, :channels_per_joint].flatten().unsqueeze(-1)
        # motion has shape [batch_size, num_joints * channels_per_joint, frames]
        # displacement has shape [batch_size, 3, frames]
        return motion, displacement
