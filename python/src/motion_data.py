import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pymotion.rotations.dual_quat as dquat
import pymotion.rotations.quat as quat
from pymotion.ops.skeleton import to_root_dual_quat


class TrainMotionData(Dataset):
    def __init__(self, param, scale, data_path, device):
        self.motions = []
        self.norm_motions = []
        self.means_motions = []
        self.var_motions = []
        self.param = param
        self.scale = scale
        self.data_path = os.path.join(data_path, "train_data.pt")
        self.data_path_temporal = os.path.join(data_path, "train_data_temporal.pt")
        self.device = device

    def add_motion(self, offsets, global_pos, rotations, parents, temporal=False):
        """
        Parameters:
        -----------
        offsets: np.array of shape (n_joints, 3)
        global_pos: np.array of shape (n_frames, 3)
        rotations: np.array of shape (n_frames, n_joints, 4) (quaternions)
        parents: np.array of shape (n_joints)

        Returns:
        --------
        self.motions:
            dqs: tensor of shape (windows_size, n_joints * 8) (dual quaternions)
        """

        # downsample
        downsample = self.param["downsample"]
        if downsample > 1:
            global_pos = global_pos[::downsample]
            rotations = rotations[::downsample]

        frames = rotations.shape[0]
        assert frames >= self.param["window_size"]
        # displacement
        global_pos = torch.from_numpy(global_pos).type(torch.float32).to(self.device)
        displacement = torch.cat(
            (torch.zeros((1, 3), device=self.device), global_pos[1:] - global_pos[:-1]),
            dim=0,
        )
        # convert displacement to root space
        displacement = quat.mul_vec(quat.inverse(rotations[:, 0, :]), displacement.cpu().numpy())
        # convert global root rotations to incremental rotations
        incr_rotations = rotations[:, 0, :].copy()
        incr_rotations[1:, :] = quat.mul(quat.inverse(rotations[:-1, 0, :]), rotations[1:, 0, :])
        incr_rotations[0, :] = np.array([1, 0, 0, 0])
        # create dual quaternions
        dqs = to_root_dual_quat(rotations, np.zeros(rotations.shape[:-2] + (3,)), parents, offsets)
        if temporal:
            # compute heights of root, hands and feet
            dqs_rots, dqs_translations = dquat.to_rotation_translation(dqs)
            dqs_translations = quat.mul_vec(dqs_rots[:, 0:1, :], dqs_translations)
            dqs_translations = torch.from_numpy(dqs_translations).to(self.device) + global_pos[:, None, :]
            heights = dqs_translations[:, self.param["height_indices"], 1].type(torch.float32)
        # root is not a dual quaternion, but a quaternion + displacement + a 0 in the last component
        # the rest of joints are dual quaternions
        dqs[..., 0, :4] = incr_rotations
        dqs = dquat.unroll(dqs, axis=0)  # ensure continuity
        dqs[..., 0, 4:7] = displacement
        dqs[..., 0, 7] = 0
        dqs = torch.from_numpy(dqs).type(torch.float32).to(self.device)
        dqs = torch.flatten(dqs, 1, 2)
        displacement = torch.from_numpy(displacement).type(torch.float32).to(self.device)
        # offsets
        offsets = torch.from_numpy(offsets).type(torch.float32).to(self.device)
        # change global_pos to (1, 3, frames)
        global_pos = global_pos.permute(1, 0).unsqueeze(0)
        if temporal:
            for start in range(0, frames, self.param["window_step"]):
                end = start + self.param["window_size"]
                if end + self.param["sample_step"] < frames:
                    # Displacement accumulated
                    displacement_window = displacement[start : end + self.param["sample_step"]]
                    displacement_past_acc = displacement_window[self.param["past_frames"]]
                    for i, start_i in enumerate(self.param["past_frames"]):
                        end_i = start_i + self.param["sample_step"]
                        displacement_past_acc[i] = torch.sum(displacement_window[start_i:end_i], dim=0)
                    # Final data
                    norm_motion = {
                        "dqs_past": dqs[start:end][self.param["past_frames"]],
                        "dqs_future": dqs[start:end][self.param["future_frames"]],
                        "displacement_past": displacement_window[self.param["past_frames"]],
                        "displacement_future": displacement_window[self.param["future_frames"]],
                    }
                    self.norm_motions.append(norm_motion)
                    motion = {
                        "offsets": offsets,
                        "displacement_past_acc": displacement_past_acc,
                        "heights": heights[start:end][self.param["past_frames"]],
                    }
                    self.motions.append(motion)
        else:
            for start in range(0, frames, self.param["window_step"]):
                end = start + self.param["window_size"]
                if end < frames:
                    motion = {
                        "dqs": dqs[start:end],
                        "displacement": displacement[start:end],
                        "offsets": offsets,
                    }
                    self.motions.append(motion)
            # Means
            motion_mean = {
                "dqs": torch.mean(dqs, dim=0).to(self.device),
                "displacement": torch.mean(displacement, dim=0).to(self.device),
            }
            self.means_motions.append(motion_mean)
            # Stds
            motion_var = {
                "dqs": torch.var(dqs, dim=0).to(self.device),
                "displacement": torch.var(displacement, dim=0).to(self.device),
            }
            self.var_motions.append(motion_var)

    def normalize(self):
        """
        Normalize motions by means and stds

        Returns:
        --------
        self.norm_motions:
            dqs: tensor of shape (windows_size, n_joints * 8) (dual quaternions)
        """
        dqs_means = torch.stack([m["dqs"] for m in self.means_motions], dim=0)
        dqs_vars = torch.stack([s["dqs"] for s in self.var_motions], dim=0)
        displacement_means = torch.stack([m["displacement"] for m in self.means_motions], dim=0)
        displacement_vars = torch.stack([s["displacement"] for s in self.var_motions], dim=0)

        self.means = {
            "dqs": torch.mean(dqs_means, dim=0).to(self.device),
            "displacement": torch.mean(displacement_means, dim=0).to(self.device),
        }

        # Source: https://stats.stackexchange.com/a/26647
        self.stds = {
            "dqs": torch.sqrt(torch.mean(dqs_vars, dim=0)).to(self.device),
            "displacement": torch.sqrt(torch.mean(displacement_vars, dim=0)).to(self.device),
        }

        if torch.count_nonzero(self.stds["dqs"]) != torch.numel(self.stds["dqs"]):
            # print("WARNING: dqs stds are zero")
            self.stds["dqs"][self.stds["dqs"] < 1e-10] = 1
        if torch.count_nonzero(self.stds["displacement"]) != torch.numel(self.stds["displacement"]):
            # print("WARNING: displacement stds are zero")
            self.stds["displacement"][self.stds["displacement"] < 1e-10] = 1

        # Normalized
        for motion in self.motions:
            norm_motion = {
                "dqs": (motion["dqs"] - self.means["dqs"]) / self.stds["dqs"],
                "displacement": (motion["displacement"] - self.means["displacement"])
                / self.stds["displacement"],
            }
            self.norm_motions.append(norm_motion)

    def normalize_force(self, means, stds):
        # Normalize
        for motion in self.norm_motions:
            motion["dqs_past"] = (motion["dqs_past"] - means["dqs"]) / stds["dqs"]
            motion["dqs_future"] = (motion["dqs_future"] - means["dqs"]) / stds["dqs"]
            motion["displacement_past"] = (motion["displacement_past"] - means["displacement"]) / stds[
                "displacement"
            ]
            motion["displacement_future"] = (motion["displacement_future"] - means["displacement"]) / stds[
                "displacement"
            ]

    def try_load(self, temporal=False):
        if temporal and os.path.exists(self.data_path_temporal):
            print("Importing temporal data from {}".format(self.data_path_temporal))
            data = torch.load(self.data_path_temporal)
        elif os.path.exists(self.data_path):
            print("Importing data from {}".format(self.data_path))
            data = torch.load(self.data_path)
        else:
            return False
        self.motions = data["motions"]
        self.norm_motions = data["norm_motions"]
        return True

    def save(self, temporal=False):
        data = {
            "motions": self.motions,
            "norm_motions": self.norm_motions,
        }
        if temporal:
            torch.save(data, self.data_path_temporal)
        else:
            torch.save(data, self.data_path)

    def __len__(self):
        return len(self.motions) - 1  # -1 because we need two motions for one sample

    def __getitem__(self, index):
        return (
            (self.motions[index], self.norm_motions[index]),
            (self.motions[index + 1], self.norm_motions[index + 1]),
        )


class TestMotionData:
    def __init__(self, param, scale, device, height_indices=None):
        self.norm_motions = []
        self.bvhs = []
        self.filenames = []
        self.param = param
        self.scale = scale
        self.device = device
        self.heights_indices = height_indices

    def set_means_stds(self, means, stds):
        self.means = means
        self.stds = stds

    def add_motion(self, offsets, global_pos, rotations, parents, bvh, filename, temporal=False):
        """
        Parameters:
        -----------
        offsets: np.array of shape (n_joints, 3)
        global_pos: np.array of shape (n_frames, 3)
        rotations: np.array of shape (n_frames, n_joints, 4) (quaternions)
        parents: np.array of shape (n_joints)

        Returns:
        --------
        self.norm_motions:
            dqs: tensor of shape (windows_size, n_joints * 8) (dual quaternions)
        """
        # downsample
        downsample = self.param["downsample"]
        if downsample > 1:
            global_pos = global_pos[::downsample]
            rotations = rotations[::downsample]

        frames = rotations.shape[0]
        assert frames >= self.param["window_size"]
        # displacement
        global_pos = torch.from_numpy(global_pos).type(torch.float32).to(self.device)
        displacement = torch.cat(
            (torch.zeros((1, 3), device=self.device), global_pos[1:] - global_pos[:-1]),
            dim=0,
        )
        # convert displacement to root space
        displacement = quat.mul_vec(quat.inverse(rotations[:, 0, :]), displacement.cpu().numpy())
        # convert global root rotations to incremental rotations
        incr_rotations = rotations[:, 0, :].copy()
        incr_rotations[1:, :] = quat.mul(quat.inverse(rotations[:-1, 0, :]), rotations[1:, 0, :])
        incr_rotations[0, :] = np.array([1, 0, 0, 0])
        # create dual quaternions
        dqs = to_root_dual_quat(rotations, np.zeros(rotations.shape[:-2] + (3,)), parents, offsets)
        if temporal or self.heights_indices is not None:
            # compute heights of root, hands and feet
            dqs_rots, dqs_translations = dquat.to_rotation_translation(dqs)
            dqs_translations = quat.mul_vec(dqs_rots[:, 0:1, :], dqs_translations)
            dqs_translations = torch.from_numpy(dqs_translations).to(self.device) + global_pos[:, None, :]
            heights = dqs_translations[
                :,
                self.heights_indices if self.heights_indices is not None else self.param["height_indices"],
                1,
            ].type(torch.float32)
        dqs[..., 0, :4] = incr_rotations
        dqs = dquat.unroll(dqs, axis=0)  # ensure continuity
        dqs[..., 0, 4:7] = displacement
        dqs[..., 0, 7] = 0
        dqs = torch.from_numpy(dqs).type(torch.float32).to(self.device)
        dqs = torch.flatten(dqs, 1, 2)
        displacement = torch.from_numpy(displacement).type(torch.float32).to(self.device)
        # offsets
        offsets = torch.from_numpy(offsets).type(torch.float32).to(self.device)
        # change global_pos to (1, 3, frames)
        global_pos = global_pos.permute(1, 0).unsqueeze(0)
        # change global_rot to (1, 4, frames)
        global_rot = (
            (torch.from_numpy(rotations[:, 0, :]).type(torch.float32).to(self.device))
            .permute(1, 0)
            .unsqueeze(0)
        )
        if temporal:
            displacement_acc = torch.zeros_like(displacement, device=self.device)
            for i in range(0, displacement.shape[0] - self.param["sample_step"]):
                displacement_acc[i] = torch.sum(displacement[i : i + self.param["sample_step"]], dim=0)

            motion = {
                "dqs": dqs,
                "displacement": displacement,
                "global_pos": global_pos,
                "global_rot": global_rot,
                "offsets": offsets,
                "displacement_acc": displacement_acc,
                "heights": heights,
            }
        else:
            motion = {
                "dqs": dqs,
                "displacement": displacement,
                "global_pos": global_pos,
                "global_rot": global_rot,
                "offsets": offsets,
            }
            if self.heights_indices is not None:
                motion["heights"] = heights
        self.norm_motions.append(motion)
        self.bvhs.append(bvh)
        self.filenames.append(filename)

    def normalize(self):
        # Normalize
        assert self.means is not None
        assert self.stds is not None
        for motion in self.norm_motions:
            motion["dqs"] = (motion["dqs"] - self.means["dqs"]) / self.stds["dqs"]
            motion["displacement"] = (motion["displacement"] - self.means["displacement"]) / self.stds[
                "displacement"
            ]

    def get_bvh(self, index):
        return self.bvhs[index], self.filenames[index]

    def get_len(self):
        return len(self.norm_motions)

    def get_item(self, index):
        return self.norm_motions[index]


class RunMotionData:
    def __init__(self, param, device):
        self.param = param
        self.device = device

    def set_means_stds(self, means, stds):
        self.means = means
        self.stds = stds

    def set_motion(self, positions, rotations):
        frames = rotations.shape[0]
        assert frames >= self.param["window_size"]
        global_pos = positions[:, 0, :]
        # displacement
        global_pos = torch.from_numpy(global_pos).type(torch.float32).to(self.device)
        displacement = torch.cat(
            (torch.zeros((1, 3), device=self.device), global_pos[1:] - global_pos[:-1]),
            dim=0,
        )
        # convert displacement to root space
        displacement = quat.mul_vec(quat.inverse(rotations[:, 0, :]), displacement.cpu().numpy())
        displacement = torch.from_numpy(displacement).type(torch.float32).to(self.device)
        # create dual quaternions
        dqs = dquat.from_rotation_translation(rotations, positions)
        dqs = dquat.unroll(dqs, axis=0)  # ensure continuity
        dqs = torch.from_numpy(dqs).type(torch.float32).to(self.device)
        dqs = torch.flatten(dqs, 1, 2)
        self.motion["dqs"] = dqs
        self.motion["displacement"] = displacement

    def normalize_motion(self):
        # Normalize
        assert self.means is not None
        assert self.stds is not None
        self.motion["dqs"] = (self.motion["dqs"] - self.means["dqs"]) / self.stds["dqs"]
        self.motion["displacement"] = (self.motion["displacement"] - self.means["displacement"]) / self.stds[
            "displacement"
        ]

    def get_item(self):
        return self.motion
