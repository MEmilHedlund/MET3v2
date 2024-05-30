import torch
from torch import nn, Tensor
from torch.nn import functional
from torch.nn import functional as F
from typing import Optional


class ContrastiveClassifier(nn.Module):
    def __init__(self, measurement_dim):
        super().__init__()
        self.measurement_dim = measurement_dim
        self.device = 'cpu'
        self.f = nn.Linear(measurement_dim, measurement_dim, bias=False)

    def forward(self, x: Tensor, padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        Expects a batch with dimensions (BATCH_SIZE x N_MEASUREMENTS x MEASUREMENT_DIM)
        """
        assert len(x.shape) == 3

        batch_size = x.shape[0]
        n_meas = x.shape[1]

        # Compute projection for each measurement and normalize them to unit hypersphere
        z = self.f(x)
        z = functional.normalize(z, dim=2)

        # Compute dot-product between all pairs (batch-wise)
        dot_products = z @ z.permute(0, 2, 1)

        # Create mask to ignore diagonal elements (dot-products between vector and itself)
        mask = (torch.eye(n_meas, device=self.device).bool()).repeat(batch_size, 1, 1)

        # Create mask to ignore dot-products between vectors and any padding elements
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1)
            temp = padding_mask.repeat(1, n_meas, 1)
            padding_mask = temp | temp.transpose(1, 2)
        else:
            padding_mask = torch.zeros(batch_size, n_meas, n_meas, dtype=torch.bool, device=self.device)

        # All elements which are masked are set to -inf, corresponding to zero probability in the predictions
        masked_dots = dot_products.masked_fill(mask | padding_mask, -100_000_000)

        probs = masked_dots.log_softmax(2)
        return probs

    def to(self, device):
        super().to(device)
        self.device = device



class ContrastiveLoss(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.device = torch.device(params.training.device)

    def forward(self, log_classifications, unique_ids) -> Tensor:
        batch_size, n_measurements = unique_ids.shape

        temp = unique_ids.unsqueeze(1).repeat(1, n_measurements, 1)
        id_matrix = (temp == temp.permute(0, 2, 1)).float()

        # Mask diagonal and then perform row-wise normalization
        mask = (torch.eye(n_measurements, device=self.device).bool()).repeat(batch_size, 1, 1)
        id_matrix = id_matrix.masked_fill(mask, 0.0)
        id_matrix = F.normalize(id_matrix, p=1, dim=2)

        # Compute element-wise multiplication between log_classifications and id_matrix (NaNs -> 0.0)
        per_measurement_losses = -log_classifications * id_matrix
        mask = torch.isnan(per_measurement_losses)
        per_measurement_losses = per_measurement_losses.masked_fill(mask, 0.0)
        per_measurement_losses = per_measurement_losses.flatten(0, 1)  # get rid of batch dimension, all measurements are born equal

        # Compute loss
        per_measurement_loss = per_measurement_losses.sum(dim=1)
        n_eligible_measurements = (per_measurement_loss != 0.0).sum()  # number of measurements with non-zero losses,
        # i.e. no. of measurements for which at least one other measurement is from the same object.
        loss = per_measurement_loss.sum()/n_eligible_measurements
        return loss
