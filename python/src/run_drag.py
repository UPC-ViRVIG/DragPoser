import os
import random
import numpy as np
import torch
from drag_pose import DragPose
from train_data import Train_Data
from generator_architecture import Generator_Model
from temporal_transformer import Temporal
from pymotion.rotations.quat import to_matrix
import train
import train_temporal


class RunDrag:
    def __init__(self):
        # Set seed
        torch.manual_seed(train.param["seed"])
        random.seed(train.param["seed"])
        np.random.seed(train.param["seed"])

        self.device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"
        # Uncomment next line to use the GPU for everything
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def log(self, msg):
        with open("log_python.txt", "a") as f:
            f.write(msg + "\n")

    def set_reference_skeleton(self, bvh_path):
        """
        bvh_path is the path to the reference skeleton
        """
        filename = os.path.basename(bvh_path)
        dir = bvh_path[: -len(filename)]
        _, _, self.parents, self.offsets, _ = train.get_info_from_bvh(train.get_bvh_from_disk(dir, filename))
        self.offsets = torch.from_numpy(self.offsets).to(self.device)
        return len(self.parents)  # num_joints

    def load_models(self, model_path):
        """
        model_path is the path to the folder containing the generator.pt and temporal.pt files
        """
        # Create Models
        self.train_data = Train_Data(self.device, train.param, None)
        self.generator_model = Generator_Model(self.device, train.param, self.parents, self.train_data).to(
            self.device
        )
        self.temporal_model = Temporal(train_temporal.param, self.device_gpu).to(self.device_gpu)

        # Load Models
        generator_model_path = os.path.join(model_path, "generator.pt")
        self.means, self.stds = train.load_model(
            self.generator_model, generator_model_path, self.train_data, self.device
        )
        temporal_model_path = os.path.join(model_path, "temporal.pt")
        self.means_latent, self.stds_latent = train_temporal.load_model(
            self.temporal_model, temporal_model_path, self.device
        )

    def set_mask_and_weights(self, mask, weights):
        """
        mask is a numpy array of shape (num_joints,)
        weights is a numpy array of shape (num_joints, 2)
        weights[:, 0] are the position weights
        weights[:, 1] are the rotation weights
        """
        assert len(mask) == len(self.parents)
        assert len(weights) == len(self.parents)
        assert weights.shape[1] == 2  # position and rotation
        self.mask = torch.from_numpy(mask).to(self.device)
        self.weights = torch.from_numpy(weights).to(self.device)
        self.mask_indices = torch.nonzero(self.mask).squeeze()
        self.weights = self.weights[self.mask_indices]
        return len(self.mask_indices)  # end effectors

    def init_drag_pose(self, initial_global_pos, initial_global_rot):
        """
        initial_global_pos is a numpy array of shape (1, 3)
        initial_global_rot is a numpy array of shape (1, 4)
        """
        self.drag = DragPose(
            self.generator_model,
            self.temporal_model,
            self.means_latent,
            self.stds_latent,
            self.device,
            self.device_gpu,
        )
        initial_pose = torch.zeros((1, len(self.parents) * 8, train.param["window_size"]), device=self.device)
        initial_global_pos = torch.from_numpy(initial_global_pos).unsqueeze(-1).to(self.device)
        initial_global_rot = torch.from_numpy(initial_global_rot).unsqueeze(-1).to(self.device)
        initial_heights = torch.zeros((6), device=self.device)
        self.drag.set_initial_pose(
            initial_pose, initial_global_pos.clone(), initial_global_rot.clone(), initial_heights
        )

    def set_optim_params(self, stop_eps_pos, stop_eps_rot, max_iter, lr):
        """
        stop_eps_pos is a float
        stop_eps_rot is a float
        max_iter is an int
        lr is a float
        """
        self.stop_eps_pos = stop_eps_pos
        self.stop_eps_rot = stop_eps_rot
        self.max_iter = max_iter
        self.learning_rate = lr

    def set_lambdas(self, lambda_rot, lambda_temporal, temporal_future_window):
        """
        lambda_rot is a float
        lambda_temporal is a float
        temporal_future_window is an int
        """
        self.lambda_rot = lambda_rot
        self.lambda_temporal = lambda_temporal
        self.temporal_future_window = temporal_future_window

    def set_global_pos(self, global_pos):
        """
        global_pos is a numpy array of shape (1, 3)
        """
        self.drag.current_global_pos = torch.from_numpy(global_pos).unsqueeze(-1).clone().to(self.device)

    def drag_pose(self, target_ee_pos, target_ee_rot, result_pose, result_global_pos):
        """
        input are numpy arrays
        target_ee_pos is a numpy array of shape (end_effectors, 3)
        target_ee_rot is a numpy array of shape (end_effectors, 4)
        output is written in result_pose and result_global_pos
        result_pose is a numpy array of shape (num_joints, 4)
        result_global_pos is a numpy array of shape (1, 3)
        """

        target_ee_rot = to_matrix(target_ee_rot)

        target_ee_pos = torch.from_numpy(target_ee_pos).to(self.device)
        target_ee_rot = torch.from_numpy(target_ee_rot).to(self.device)

        res_pose, res_global_pos = self.drag.run(
            target_ee_pos=target_ee_pos,
            target_ee_rot=target_ee_rot,
            mask_joints=self.mask_indices,
            weights_joints=self.weights,
            offsets=self.offsets,
            stop_eps_pos=self.stop_eps_pos,
            stop_eps_rot=self.stop_eps_rot,
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            lambda_rot=self.lambda_rot,
            lambda_temporal=self.lambda_temporal,
            temporal_future_window=self.temporal_future_window,
            height_indices=[0, 4, 8, 13, 17, 21],
            joint_adjustment_indices=None,
            verbose=False,
        )
        # res_pose is a tensor of shape (1, num_joints * 4, 1)
        # res_global_pos is a tensor of shape (1, 3, 1)

        res_pose = (
            res_pose * self.stds["dqs"].reshape((-1, 8))[:, :4].flatten()
            + self.means["dqs"].reshape((-1, 8))[:, :4].flatten()
        )
        res_pose = res_pose.reshape(1, -1, 4).detach().cpu().numpy()
        rots = train.from_root_quat(res_pose, self.parents)

        # Convert results to NumPy arrays and update the passed-in arrays
        rots_numpy = rots.squeeze().reshape(-1, 4).copy()
        res_global_pos_numpy = res_global_pos.squeeze().detach().cpu().numpy().copy()

        # Update result_pose array
        np.copyto(result_pose, rots_numpy)

        # Update result_global_pos array
        result_global_pos[0, :] = res_global_pos_numpy
