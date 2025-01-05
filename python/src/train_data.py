import torch


class Train_Data:
    def __init__(self, device, param, args):
        super().__init__()
        self.device = device
        self.param = param
        if args is None:
            self.use_fk_loss = False
        else:
            self.use_fk_loss = args.fk
        self.losses = []

    def set_motions(
        self, offsets, dqs, displacement, next_dqs=None, next_displacement=None
    ):
        # swap second and third dimensions for convolutions (last row should be time)
        if next_dqs is None:
            self.motion = dqs.permute(0, 2, 1)
            self.displacement = displacement.permute(0, 2, 1)
        else:
            self.motion = torch.cat(
                (
                    dqs.permute(0, 2, 1).unsqueeze(1),
                    next_dqs.permute(0, 2, 1).unsqueeze(1),
                ),
                dim=1,
            )
            self.displacement = torch.cat(
                (
                    displacement.permute(0, 2, 1).unsqueeze(1),
                    next_displacement.permute(0, 2, 1).unsqueeze(1),
                ),
                dim=1,
            )
        for loss in self.losses:
            loss.set_offsets(offsets)
            loss.use_fk(self.use_fk_loss)

    def set_means(self, mean_dqs, mean_displacement):
        self.mean_dqs = mean_dqs
        self.mean_displacement = mean_displacement
        for loss in self.losses:
            loss.set_mean(mean_dqs)

    def set_stds(self, std_dqs, std_displacement):
        self.std_dqs = std_dqs
        self.std_displacement = std_displacement
        for loss in self.losses:
            loss.set_std(std_dqs)
