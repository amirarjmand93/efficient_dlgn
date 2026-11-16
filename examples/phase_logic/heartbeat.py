"""MIT-BIH heartbeat encoding for the existing CompactLogic backbone."""

import torch


class HeartbeatQ2LocalEncoder(torch.nn.Module):
    """Two training-set quantile bits plus two adjacent-sample comparisons.

    The 187 samples are encoded in temporal order, then reshaped to 11 x 17
    for the existing 2D CompactLogic convolutions. No learned preprocessing is
    introduced; thresholds are persistent model buffers.
    """

    output_channels = 4
    num_bits = 4
    num_quantile_bits = 2
    num_lbp_bits = 2
    num_sine_bits = 0
    channel_order = "quantile_1, quantile_2, center>left, center>right"

    def __init__(self, thresholds: torch.Tensor):
        super().__init__()
        values = torch.as_tensor(thresholds, dtype=torch.float32).flatten()
        if values.shape != (2,) or not torch.isfinite(values).all():
            raise ValueError("Heartbeat Q2 encoder requires two finite thresholds.")
        if values[0] > values[1]:
            raise ValueError("Heartbeat quantile thresholds must be ordered.")
        self.register_buffer("thresholds", values.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (1, 187):
            raise ValueError("Heartbeat encoder expects [batch, 1, 187].")
        signal = x[:, 0]
        left = torch.cat((signal[:, :1], signal[:, :-1]), dim=1)
        right = torch.cat((signal[:, 1:], signal[:, -1:]), dim=1)
        channels = torch.stack((
            signal > self.thresholds[0],
            signal > self.thresholds[1],
            signal > left,
            signal > right,
        ), dim=1)
        return channels.reshape(x.shape[0], 4, 11, 17).to(x.dtype)
