from typing import Sequence

import torch
from torch import nn


class RotationInvariantMLP(nn.Module):
    """TODO: docstring"""
    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1)) -> None:
        super().__init__()

        assert len(mlp_features) > 0
        mlp_layers = []
        for out_features in mlp_features:
            mlp_layers.extend([
                nn.Linear(in_features, out_features),
                nn.ReLU(inplace=True),
            ])
            in_features = out_features
        self.mlp = nn.Sequential(*mlp_layers)

        assert pooling in ["max", "mean"], f"Unsupported pooling: {pooling}"
        self.pooling = pooling

        self.offsets = offsets if len(offsets) > 0 else (0, )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # (T, N, ...)

        # Create a new dim for band rotation augmentation with each entry
        # corresponding to the original tensor with its channels/electrodes
        # shifted by one of ``offsets``:
        # (T, N, ...) -> (T, N, rotation, ...)
        x = torch.stack([x.roll(offset, dims=2) for offset in self.offsets],
                        dim=2)

        # Flatten features and pass through MLP:
        # (T, N, rotation, ...) -> (T, N, rotation, mlp_features[-1])
        x = self.mlp(x.flatten(start_dim=3))

        # Pool over rotations:
        # (T, N, rotation, mlp_features[-1]) -> (T, N, mlp_features[-1])
        return x.max(dim=2).values if self.pooling == "max" else x.mean(dim=2)


class LeftRightRotationInvariantMLP(nn.Module):
    """TODO docstring: (T, N, band, ...)"""
    def __init__(self,
                 in_features: int,
                 mlp_features: Sequence[int],
                 pooling: str = "mean",
                 offsets: Sequence[int] = (-1, 0, 1),
                 stack_dim: int = 2) -> None:
        super().__init__()

        self.left_mlp = RotationInvariantMLP(in_features,
                                             mlp_features,
                                             pooling=pooling,
                                             offsets=offsets)
        self.right_mlp = RotationInvariantMLP(in_features,
                                              mlp_features,
                                              pooling=pooling,
                                              offsets=offsets)

        self.stack_dim = stack_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[self.stack_dim] == 2
        left, right = inputs.unbind(self.stack_dim)

        left = self.left_mlp(left)
        right = self.right_mlp(right)

        return torch.stack([left, right], dim=self.stack_dim)


class TDSConv2dBlock(nn.Module):
    """TODO: docstring"""
    def __init__(self, channels: int, width: int, kernel_width: int) -> None:
        super().__init__()

        self.conv2d = nn.Conv2d(channels,
                                channels,
                                kernel_size=(1, kernel_width))
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(channels * width)

        self.channels = channels
        self.width = width

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T_in, N, C = inputs.shape  # TNC

        # TNC -> NCT -> NcwT
        x = inputs.movedim(0, -1).view(N, self.channels, self.width, T_in)
        x = self.conv2d(x)
        x = self.relu(x)
        x = x.view(N, C, -1).movedim(-1, 0)  # NcwT -> NCT -> TNC

        # Skip connection after downsampling
        T_out = x.shape[0]
        x += inputs[-T_out:]

        # Layer norm over C
        return self.layer_norm(x)  # TNC


class TDSFullyConnectedBlock(nn.Module):
    """TODO: docstring"""
    def __init__(self, num_features: int) -> None:
        super().__init__()

        self.fc_block = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(inplace=True),
            nn.Linear(num_features, num_features),
        )
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # TNC
        x = self.fc_block(x)
        x += inputs
        return self.layer_norm(x)  # TNC


class TDSConvEncoder(nn.Module):
    """TODO: docstring"""
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 block_channels: Sequence[int] = (24, 24, 24, 24),
                 kernel_width: int = 11) -> None:
        super().__init__()

        self.rotation_invariant_mlp = LeftRightRotationInvariantMLP(
            in_features, [out_features // 2], pooling="mean")

        assert len(block_channels) > 0
        tds_conv_blocks = []
        for channels in block_channels:
            assert out_features % channels == 0, (
                "block_channels must evenly divide out_features")
            tds_conv_blocks.extend([
                TDSConv2dBlock(channels, out_features // channels,
                               kernel_width),
                TDSFullyConnectedBlock(out_features),
            ])
        self.tds_conv_blocks = nn.Sequential(*tds_conv_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # (T, N, bands=2, freq, channels=16)
        x = self.rotation_invariant_mlp(x)  # (T, N, bands=2, out_features/2)
        x = x.flatten(start_dim=2)  # (T, N, out_features)
        return self.tds_conv_blocks(x)  # (T, N, out_features)
