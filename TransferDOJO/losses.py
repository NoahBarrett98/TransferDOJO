import torch.nn as nn
import torch


class NT_Xent(nn.Module):
    """
    normalized temperature-scaled cross entropy loss
    """

    def __init__(self, batch_size, temperature):
        self.batch_size = batch_size
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        implementation adapted from https://github.com/leftthomas/SimCLR
        :param z_i: image latent 1
        :param z_j: image latent 2
        :return:
        """
        # [2*B, D]
        out = torch.cat([z_i, z_j], dim=0)
        # sim(z_i, z_j)/t
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)
        # matrix for 1 E {0,1} function, e.g.
        # 0 1 1
        # 1 0 1
        # 1 1 0
        mask = (
            torch.ones_like(sim_matrix)
            - torch.eye(2 * self.batch_size, device=sim_matrix.device)
        ).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * self.batch_size, -1)

        # compute loss
        pos_sim = torch.exp(torch.sum(z_i * z_j, dim=-1) / self.temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)

        # loss
        return (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
