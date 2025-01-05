import torch
import torch.nn as nn
import pymotion.rotations.quat_torch as quat
from pymotion.ops.forward_kinematics_torch import fk


def from_root_quat(q: torch.Tensor, parents: torch.Tensor) -> torch.Tensor:
    """
    Convert root-centered quaternions to the skeleton information.

    Parameters
    ----------
    dq: torch.Tensor[..., n_joints, 4]
        Includes as first element the global position of the root joint
    parents: torch.Tensor[n_joints]

    Returns
    -------
    rotations : torch.Tensor[..., n_joints, 4]
    """
    n_joints = q.shape[-2]
    # rotations has shape (frames, n_joints, 4)
    rotations = q.clone()
    # make transformations local to the parents
    # (initially local to the root)
    for j in reversed(range(1, n_joints)):
        parent = parents[j]
        if parent == 0:  # already in root space
            continue
        inv = quat.inverse(rotations[..., parent, :])
        rotations[..., j, :] = quat.mul(inv, rotations[..., j, :])
    return rotations


class MSE_DQ(nn.Module):
    def __init__(self, param, parents, device) -> None:
        super().__init__()
        self.mse = nn.MSELoss()
        self.param = param
        self.parents = parents
        self.device = device
        self.channels_per_joint = 4

    def set_mean(self, mean_dqs):
        self.mean_dqs = (
            mean_dqs.clone()
            .reshape((-1, 8))[:, : self.channels_per_joint]
            .flatten()
            .unsqueeze(-1)
        )

    def set_std(self, std_dqs):
        self.std_dqs = (
            std_dqs.clone()
            .reshape((-1, 8))[:, : self.channels_per_joint]
            .flatten()
            .unsqueeze(-1)
        )

    def set_offsets(self, offsets):
        self.offsets = offsets.unsqueeze(1).unsqueeze(1)
        self.offsets = torch.tile(self.offsets, (1, 2, 1, 1, 1))

    def use_fk(self, use_fk_loss):
        self.use_fk_loss = use_fk_loss

    def forward_generator(
        self,
        input_motion,
        input_displacement,
        target_motion,
        target_displacement,
        mu,
        logvar,
        latent,
    ):
        qs = input_motion.clone()
        target_dqs = target_motion.clone()
        if target_dqs.ndim == 3:
            target_qs = target_dqs.reshape(
                (target_dqs.shape[0], -1, 8, target_dqs.shape[-1])
            )[..., :4, :].reshape((target_dqs.shape[0], -1, target_dqs.shape[-1]))
        else:
            target_qs = target_dqs.reshape(
                (target_dqs.shape[0], target_dqs.shape[1], -1, 8, target_dqs.shape[-1])
            )[..., :4, :].reshape(
                (target_dqs.shape[0], target_dqs.shape[1], -1, target_dqs.shape[-1])
            )

        # Quaternions MSE Loss
        loss_joints = self.mse(qs[..., 4:, :], target_qs[..., 4:, :])
        loss_root = self.mse(qs[..., :4, :], target_qs[..., :4, :])

        # FK Loss
        if self.use_fk_loss:
            qs = qs * self.std_dqs + self.mean_dqs
            target_qs = target_qs * self.std_dqs + self.mean_dqs
            qs[..., :4, :] = torch.tensor([1, 0, 0, 0], device=self.device).unsqueeze(
                -1
            )
            target_qs[..., :4, :] = torch.tensor(
                [1, 0, 0, 0], device=self.device
            ).unsqueeze(-1)
            qs_local = from_root_quat(
                qs.permute(0, 1, 3, 2).reshape(
                    (
                        qs.shape[0],
                        qs.shape[1],
                        qs.shape[-1],
                        -1,
                        self.channels_per_joint,
                    )
                ),
                self.parents,
            )
            target_qs_local = from_root_quat(
                target_qs.permute(0, 1, 3, 2).reshape(
                    (
                        target_qs.shape[0],
                        target_qs.shape[1],
                        target_qs.shape[-1],
                        -1,
                        self.channels_per_joint,
                    )
                ),
                self.parents,
            )
            pos_qs, _ = fk(
                qs_local,
                torch.zeros((3,), device=self.device),
                self.offsets,
                self.parents,
            )
            pos_target_qs, _ = fk(
                target_qs_local,
                torch.zeros((3,), device=self.device),
                self.offsets,
                self.parents,
            )
            loss_fk = self.mse(pos_qs, pos_target_qs)

        # Displacement MSE Loss
        loss_displacement = self.mse(input_displacement, target_displacement)

        # KLD Loss
        loss_kld = -0.5 * torch.mean(
            torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).flatten(), dim=0
        )

        # Consecutive Loss
        latent_pairs = latent.reshape(latent.shape[0] // 2, 2, -1)
        # L_consecutive = mse(z_0 - (D(z_0) - p_1) * grad(D(z_0)), z_1)
        # f = (D(z_0) - p_1) ** 2
        # L_consecutive = mse(z_0 - grad(f), z_1)
        # f = p0 - p1
        f = torch.sum((pos_qs[:, 0, ...] - pos_qs[:, 1, ...]) ** 2, dim=[1, 2, 3])
        # grad(f) wrt latent
        grad_f = torch.autograd.grad(torch.unbind(f, dim=0), latent, create_graph=True)[
            0
        ]
        # z_0 - grad(f)
        z_drag = (
            latent_pairs[:, 0, :] - grad_f.reshape(latent.shape[0] // 2, 2, -1)[:, 0, :]
        )
        # mse(z_0 - grad(f), z_1)
        loss_consecutive = self.mse(z_drag, latent_pairs[:, 1, :])

        if self.use_fk_loss:
            return (
                ("kld", loss_kld * self.param["lambda_kld"]),
                ("root", loss_root * self.param["lambda_root"]),
                ("displacement", loss_displacement * self.param["lambda_displacement"]),
                ("consecutive", loss_consecutive * self.param["lambda_consecutive"]),
                ("fk", loss_fk * self.param["lambda_fk"]),
                ("joints", loss_joints),
            )
        else:
            return (
                ("kld", loss_kld * self.param["lambda_kld"]),
                ("root", loss_root * self.param["lambda_root"]),
                ("displacement", loss_displacement * self.param["lambda_displacement"]),
                ("consecutive", loss_consecutive * self.param["lambda_consecutive"]),
                ("joints", loss_joints),
            )
