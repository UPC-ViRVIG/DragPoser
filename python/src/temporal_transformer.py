import torch
import torch.nn as nn
from positional_encoding import PositionalEncoding


class Temporal(nn.Module):
    def __init__(self, param, device):
        super().__init__()
        self.param = param
        self.device = device
        self.n_features = param["features_transformer"]
        self.max_len = len(param["past_frames"]) + len(param["future_frames"])

        # Layers
        self.in_dropout = nn.Dropout(param["dropout"])
        self.positional_encoding = PositionalEncoding(
            self.n_features,
            param["dropout"],
            self.max_len,
        )
        # + 3 for the accumulated displacement
        # + heights
        additional_input_dim = 3 + len(param["height_indices"])
        self.in_proj_encoder = nn.Linear(param["latent_dim"] + additional_input_dim, self.n_features)
        self.in_proj_decoder = nn.Linear(param["latent_dim"], self.n_features)
        self.temporal = nn.Transformer(
            d_model=self.n_features,
            nhead=param["n_heads"],
            num_encoder_layers=param["n_encoder_layers"],
            num_decoder_layers=param["n_decoder_layers"],
            dim_feedforward=param["dim_feedforward"],
            dropout=param["dropout"],
        )
        self.out_proj = nn.Linear(self.n_features, param["latent_dim"])

        parameters = (
            list(self.temporal.parameters())
            + list(self.in_proj_encoder.parameters())
            + list(self.in_proj_decoder.parameters())
            + list(self.out_proj.parameters())
        )
        self.model_parameters = parameters

        # Print number parameters
        dec_params = 0
        for parameter in parameters:
            dec_params += parameter.numel()
        print("# parameters temporal:", dec_params)

        self.optimizer = torch.optim.Adam(parameters, param["learning_rate"])
        self.loss = nn.MSELoss().to(device)

    def forward(self, latent, latent_target, tgt_mask=None):
        # latent has shape (batch_size, past samples, latent_dim)
        # latent_target has shape (batch_size, future samples, latent_dim)

        latent = self.in_dropout(latent)
        # latent_target = self.in_dropout(latent_target)

        latent = self.in_proj_encoder(latent)
        latent_target = self.in_proj_decoder(latent_target)

        # latent has shape (batch_size, past samples, features)
        # latent_target has shape (batch_size, future samples, features)

        latent = self.positional_encoding(latent)
        latent_target = self.positional_encoding(latent_target)

        # change to shape (past samples, batch_size, features)
        latent = latent.permute(1, 0, 2)
        # change to shape (future samples, batch_size, features)
        latent_target = latent_target.permute(1, 0, 2)

        self.output = self.temporal(latent, latent_target, tgt_mask=tgt_mask)
        # output has shape (future samples, batch_size, features)
        self.output = self.out_proj(self.output).permute(1, 0, 2)
        # output has shape (batch_size, future samples, latent_dim)
        return self.output

    def optimize_parameters(self, target_latent):
        self.optimizer.zero_grad()

        loss = self.compute_loss(target_latent)
        loss.backward()

        self.optimizer.step()
        return loss.item()

    def compute_loss(self, target_latent):
        # target_latent has shape (batch_size, future samples, latent_dim)
        loss = self.loss(self.output, target_latent)
        return loss

    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1)  # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float("-inf"))  # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0))  # Convert ones to 0

        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]

        return mask
