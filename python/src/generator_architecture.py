import torch
import torch.nn as nn
from autoencoder import Autoencoder
from loss import MSE_DQ


class Generator_Model(nn.Module):
    def __init__(self, device, param, parents, train_data) -> None:
        super().__init__()

        self.device = device
        self.param = param
        self.parents = parents
        self.data = train_data

        self.autoencoder = Autoencoder(param, parents, device).to(device)

        parameters = list(self.autoencoder.parameters())
        self.model_parameters = parameters

        # Print number parameters
        dec_params = 0
        for parameter in parameters:
            dec_params += parameter.numel()
        print("# parameters generator:", dec_params)

        self.optimizer = torch.optim.AdamW(parameters, param["learning_rate"])

        self.loss = MSE_DQ(param, parents, device).to(device)
        train_data.losses.append(self.loss)

    def forward(self):
        # Execute Autoencoder to obtain the motion
        (
            self.res_decoder_motion,
            self.res_decoder_displacement,
            self.mu,
            self.logvar,
            self.latent,
        ) = self.autoencoder(
            self.data.motion,
            self.data.mean_dqs,
            self.data.std_dqs,
        )
        return self.res_decoder_motion, self.res_decoder_displacement

    def compute_losses(self):
        losses = self.loss.forward_generator(
            self.res_decoder_motion,
            self.res_decoder_displacement,
            self.data.motion,
            self.data.displacement,
            self.mu,
            self.logvar,
            self.latent,
        )
        loss = sum([loss[1] for loss in losses])
        # detach losses from graph
        losses = [(loss[0], loss[1].item()) for loss in losses]
        return loss, losses

    def optimize_parameters(self):
        self.optimizer.zero_grad()

        loss, losses = self.compute_losses()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model_parameters, self.param["clip_grad_value"])

        self.optimizer.step()
        return loss, losses

    def sample(self, n_samples, mean=None):
        # latent = torch.randn(
        #     (n_samples, self.param["latent_dim"]),
        #     device=self.autoencoder.decoder.device,
        # )

        base_std = 0.3

        size = (n_samples, self.param["latent_dim"])
        if mean is None:
            mean = torch.zeros(size, device=self.autoencoder.decoder.device)
        else:
            mean = torch.tile(mean[0], (n_samples, 1))
        # mean[:, 20:100] = 1.0
        std = torch.ones(mean.shape, device=mean.device) * base_std

        # Generate normally distributed tensor
        latent = torch.normal(mean, std).to(mean.device)

        res_motion, res_displacement = self.autoencoder.decoder(latent, self.data.mean_dqs, self.data.std_dqs)
        return res_motion, res_displacement
