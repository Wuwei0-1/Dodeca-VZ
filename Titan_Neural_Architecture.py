import torch
import torch.nn as nn


class TitanBrain(nn.Module):
    """Torch implementation of the Titan controller used by the repo scripts.

    The project originally referenced a larger architecture, but the scripts only
    require a small, reliable model that accepts a batch of sensor packets and
    produces a control vector. This implementation keeps the public API used by
    the training script:

        outputs, _ = titan(inputs, None)

    where ``inputs`` has shape ``(batch, time_steps, features)``.
    """

    def __init__(self, input_dim=8, hidden_dim=64, seq_len=10, output_dim=4):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.output_dim = output_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x, context=None):
        if x.dim() == 2:
            x = x.unsqueeze(1).repeat(1, self.seq_len, 1)

        if x.dim() != 3:
            raise ValueError(f"TitanBrain expected a 3D tensor, got shape {tuple(x.shape)}")

        batch_size, seq_len, feature_dim = x.shape
        if feature_dim != self.input_dim:
            raise ValueError(
                f"Expected input features={self.input_dim}, got {feature_dim}. "
                f"Shape was {tuple(x.shape)}"
            )

        encoded = self.encoder(x.reshape(-1, feature_dim)).reshape(batch_size, seq_len, -1)
        summary = encoded.mean(dim=1)
        output = self.head(summary)
        return output, summary


__all__ = ["TitanBrain"]
