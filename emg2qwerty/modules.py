from typing import Sequence

import torch
from torch import nn


class SpectrogramNorm(nn.Module):
    """TODO: docstring - channels -> total electrode channels"""

    def __init__(self, channels: int) -> None:
        super().__init__()

        self.batch_norm = nn.BatchNorm2d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # (T, N, bands=2, frequency_bins, electrode_channels=16)
        T, N, bands, freq, C = inputs.shape

        # (T, N, bands=2, freq, C=16) -> (N, C=16, bands=2, freq, T)
        x = inputs.transpose(0, 1).transpose(1, -1)

        # Normalize spectrogram per electrode channel per band. With left
        # and right bands and 16 electrode channels per band, spectrograms
        # corresponding to each of the 2 * 16 = 32 channels are normalized
        # independently using `nn.BatchNorm2d` such that stats are computed
        # over (N, freq, time) slices.
        x = x.reshape(N, C * bands, freq, T)
        x = self.batch_norm(x)
        x = x.reshape(N, C, bands, freq, T)

        # (N, C=16, bands=2, freq, T) -> (T, N, bands=2, freq, C=16)
        return x.transpose(1, -1).transpose(0, 1)


class RotationInvariantMLP(nn.Module):
    """TODO: docstring"""

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
    ) -> None:
        super().__init__()

        assert len(mlp_features) > 0
        mlp_layers = []
        for out_features in mlp_features:
            mlp_layers.extend(
                [
                    nn.Linear(in_features, out_features),
                    nn.ReLU(inplace=True),
                ]
            )
            in_features = out_features
        self.mlp = nn.Sequential(*mlp_layers)

        assert pooling in ["max", "mean"], f"Unsupported pooling: {pooling}"
        self.pooling = pooling

        self.offsets = offsets if len(offsets) > 0 else (0,)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # (T, N, ...)

        # Create a new dim for band rotation augmentation with each entry
        # corresponding to the original tensor with its electrode channels
        # shifted by one of ``offsets``:
        # (T, N, ...) -> (T, N, rotation, ...)
        x = torch.stack([x.roll(offset, dims=2) for offset in self.offsets], dim=2)

        # Flatten features and pass through MLP:
        # (T, N, rotation, ...) -> (T, N, rotation, mlp_features[-1])
        x = self.mlp(x.flatten(start_dim=3))

        # Pool over rotations:
        # (T, N, rotation, mlp_features[-1]) -> (T, N, mlp_features[-1])
        return x.max(dim=2).values if self.pooling == "max" else x.mean(dim=2)


class MultiBandRotationInvariantMLP(nn.Module):
    """TODO docstring: (T, N, band, ...)"""

    def __init__(
        self,
        num_bands: int,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
        stack_dim: int = 2,
    ) -> None:
        super().__init__()

        self.num_bands = num_bands
        self.stack_dim = stack_dim

        # Separate MLP for each band
        self.mlps = nn.ModuleList()
        for i in range(num_bands):
            self.mlps.append(
                RotationInvariantMLP(
                    in_features, mlp_features, pooling=pooling, offsets=offsets
                )
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[self.stack_dim] == self.num_bands

        inputs_per_band = inputs.unbind(self.stack_dim)
        outputs_per_band = [
            mlp(input) for mlp, input in zip(self.mlps, inputs_per_band)
        ]
        return torch.stack(outputs_per_band, dim=self.stack_dim)


class TDSConv2dBlock(nn.Module):
    """TODO: docstring"""

    def __init__(self, channels: int, width: int, kernel_width: int) -> None:
        super().__init__()

        self.conv2d = nn.Conv2d(
            in_channels=channels, out_channels=channels, kernel_size=(1, kernel_width)
        )
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(channels * width)

        self.channels = channels
        self.width = width

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T_in, N, C = inputs.shape  # TNC

        # TNC -> NCT -> NcwT
        x = inputs.movedim(0, -1).reshape(N, self.channels, self.width, T_in)
        x = self.conv2d(x)
        x = self.relu(x)
        x = x.reshape(N, C, -1).movedim(-1, 0)  # NcwT -> NCT -> TNC

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

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
    ) -> None:
        super().__init__()

        assert len(block_channels) > 0
        tds_conv_blocks = []
        for channels in block_channels:
            assert (
                num_features % channels == 0
            ), "block_channels must evenly divide num_features"
            tds_conv_blocks.extend(
                [
                    TDSConv2dBlock(channels, num_features // channels, kernel_width),
                    TDSFullyConnectedBlock(num_features),
                ]
            )
        self.tds_conv_blocks = nn.Sequential(*tds_conv_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tds_conv_blocks(inputs)  # (T, N, num_features)
