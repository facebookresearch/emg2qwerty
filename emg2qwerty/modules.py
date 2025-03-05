# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence

import torch
from torch import nn


class SpectrogramNorm(nn.Module):
    """A `torch.nn.Module` that applies 2D batch normalization over spectrogram
    per electrode channel per band. Inputs must be of shape
    (T, N, num_bands, electrode_channels, frequency_bins).

    With left and right bands and 16 electrode channels per band, spectrograms
    corresponding to each of the 2 * 16 = 32 channels are normalized
    independently using `nn.BatchNorm2d` such that stats are computed
    over (N, freq, time) slices.

    Args:
        channels (int): Total number of electrode channels across bands
            such that the normalization statistics are calculated per channel.
            Should be equal to num_bands * electrode_chanels.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

        self.batch_norm = nn.BatchNorm2d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T, N, bands, C, freq = inputs.shape  # (T, N, bands=2, C=16, freq)
        assert self.channels == bands * C

        x = inputs.movedim(0, -1)  # (N, bands=2, C=16, freq, T)
        x = x.reshape(N, bands * C, freq, T)
        x = self.batch_norm(x)
        x = x.reshape(N, bands, C, freq, T)
        return x.movedim(-1, 0)  # (T, N, bands=2, C=16, freq)


class RotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that takes an input tensor of shape
    (T, N, electrode_channels, ...) corresponding to a single band, applies
    an MLP after shifting/rotating the electrodes for each positional offset
    in ``offsets``, and pools over all the outputs.

    Returns a tensor of shape (T, N, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input of
            shape (T, N, C, ...), this should be equal to C * ... (that is,
            the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
    ) -> None:
        super().__init__()

        assert len(mlp_features) > 0
        mlp: list[nn.Module] = []
        for out_features in mlp_features:
            mlp.extend(
                [
                    nn.Linear(in_features, out_features),
                    nn.ReLU(),
                ]
            )
            in_features = out_features
        self.mlp = nn.Sequential(*mlp)

        assert pooling in {"max", "mean"}, f"Unsupported pooling: {pooling}"
        self.pooling = pooling

        self.offsets = offsets if len(offsets) > 0 else (0,)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # (T, N, C, ...)

        # Create a new dim for band rotation augmentation with each entry
        # corresponding to the original tensor with its electrode channels
        # shifted by one of ``offsets``:
        # (T, N, C, ...) -> (T, N, rotation, C, ...)
        x = torch.stack([x.roll(offset, dims=2) for offset in self.offsets], dim=2)

        # Flatten features and pass through MLP:
        # (T, N, rotation, C, ...) -> (T, N, rotation, mlp_features[-1])
        x = self.mlp(x.flatten(start_dim=3))

        # Pool over rotations:
        # (T, N, rotation, mlp_features[-1]) -> (T, N, mlp_features[-1])
        if self.pooling == "max":
            return x.max(dim=2).values
        else:
            return x.mean(dim=2)


class MultiBandRotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that applies a separate instance of
    `RotationInvariantMLP` per band for inputs of shape
    (T, N, num_bands, electrode_channels, ...).

    Returns a tensor of shape (T, N, num_bands, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input
            of shape (T, N, num_bands, C, ...), this should be equal to
            C * ... (that is, the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
        num_bands (int): ``num_bands`` for an input of shape
            (T, N, num_bands, C, ...). (default: 2)
        stack_dim (int): The dimension along which the left and right data
            are stacked. (default: 2)
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
        num_bands: int = 2,
        stack_dim: int = 2,
    ) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.stack_dim = stack_dim

        # One MLP per band
        self.mlps = nn.ModuleList(
            [
                RotationInvariantMLP(
                    in_features=in_features,
                    mlp_features=mlp_features,
                    pooling=pooling,
                    offsets=offsets,
                )
                for _ in range(num_bands)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[self.stack_dim] == self.num_bands

        inputs_per_band = inputs.unbind(self.stack_dim)
        outputs_per_band = [mlp(_input) for mlp, _input in zip(self.mlps, inputs_per_band)]
        return torch.stack(outputs_per_band, dim=self.stack_dim)


class TDSConv2dBlock(nn.Module):
    """A 2D temporal convolution block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        channels (int): Number of input and output channels. For an input of
            shape (T, N, num_features), the invariant we want is
            channels * width = num_features.
        width (int): Input width. For an input of shape (T, N, num_features),
            the invariant we want is channels * width = num_features.
        kernel_width (int): The kernel size of the temporal convolution.
    """

    def __init__(self, channels: int, width: int, kernel_width: int) -> None:
        super().__init__()
        self.channels = channels
        self.width = width

        self.conv2d = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_width),
        )
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(channels * width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T_in, N, C = inputs.shape  # TNC

        # TNC -> NCT -> NcwT
        x = inputs.movedim(0, -1).reshape(N, self.channels, self.width, T_in)
        x = self.conv2d(x)
        x = self.relu(x)
        x = x.reshape(N, C, -1).movedim(-1, 0)  # NcwT -> NCT -> TNC

        # Skip connection after downsampling
        T_out = x.shape[0]
        x = x + inputs[-T_out:]

        # Layer norm over C
        return self.layer_norm(x)  # TNC


class MultiScaleTDSConv2dBlock(nn.Module):
    """A Multi-Scale Time-Depth Separable 2D Convolution Block.

    This extends the standard TDS block by applying multiple parallel
    convolutions with different kernel sizes to capture various time-scale dependencies.

    Args:
        channels (int): Number of input and output channels.
        width (int): Feature width (channels * width = num_features).
        kernel_widths (list of int): List of kernel sizes for multi-scale convolutions.
    """

    def __init__(
        self, channels: int, width: int, kernel_widths: Sequence[int] = (16, 32, 64)
    ) -> None:
        super().__init__()
        self.channels = channels
        self.width = width

        # Multiple parallel temporal convolutions
        self.conv2d_branches = nn.ModuleList(
            [
                nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(1, k))
                for k in kernel_widths
            ]
        )

        # 1x1 convolution to merge multi-scale features back to original shape
        self.merge_conv = nn.Conv2d(
            in_channels=len(kernel_widths) * channels, out_channels=channels, kernel_size=1
        )

        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(channels * width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T_in, N, C = inputs.shape  # TNC

        # Reshape for 2D convolutions: TNC -> NCT -> NCHW (H=width, W=T)
        x = inputs.movedim(0, -1).reshape(N, self.channels, self.width, T_in)

        # Apply multi-scale convolutions
        multi_scale_features = [conv(x) for conv in self.conv2d_branches]

        # Find the minimum time dimension (width) among all features
        min_time_dim = min(feat.size(3) for feat in multi_scale_features)

        # Trim all features to the minimum time dimension
        multi_scale_features = [feat[..., :min_time_dim] for feat in multi_scale_features]

        # Concatenate along the channel dimension
        x = torch.cat(multi_scale_features, dim=1)

        # Merge features using 1x1 convolution
        x = self.merge_conv(x)
        x = self.relu(x)

        # Reshape back: NCHW -> NCT -> TNC
        x = x.reshape(N, C, -1).movedim(-1, 0)

        # Residual connection (skip connection)
        T_out = x.shape[0]
        x = x + inputs[-T_out:]

        # Layer normalization over channels
        return self.layer_norm(x)  # TNC


class TDSFullyConnectedBlock(nn.Module):
    """A fully connected block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()

        self.fc_block = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features),
        )
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # TNC
        x = self.fc_block(x)
        x = x + inputs
        return self.layer_norm(x)  # TNC


class TDSConvEncoder(nn.Module):
    """A time depth-separable convolutional encoder composing a sequence
    of `TDSConv2dBlock` and `TDSFullyConnectedBlock` as per
    "Sequence-to-Sequence Speech Recognition with Time-Depth Separable
    Convolutions, Hannun et al" (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        block_channels (list): A list of integers indicating the number
            of channels per `TDSConv2dBlock`.
        kernel_width (int): The kernel size of the temporal convolutions.
    """

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
        multi_scale: bool = False,
    ) -> None:
        super().__init__()
        self.multi_scale = multi_scale
        kernel_widths = (kernel_width // 2, kernel_width, kernel_width * 2)
        assert len(block_channels) > 0
        tds_conv_blocks: list[nn.Module] = []
        for channels in block_channels:
            assert num_features % channels == 0, "block_channels must evenly divide num_features"

            if self.multi_scale:
                tds_conv_blocks.extend(
                    [
                        MultiScaleTDSConv2dBlock(channels, num_features // channels, kernel_widths),
                        TDSFullyConnectedBlock(num_features),
                    ]
                )
            else:
                tds_conv_blocks.extend(
                    [
                        TDSConv2dBlock(channels, num_features // channels, kernel_width),
                        TDSFullyConnectedBlock(num_features),
                    ]
                )
        self.tds_conv_blocks = nn.Sequential(*tds_conv_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tds_conv_blocks(inputs)  # (T, N, num_features)


class EMGSpecAutoEncoder(nn.Module):
    """An autoencoder that compresses EMG spectrograms from 32 channels (2 bands × 16 electrodes)
    to 16 channels and can reconstruct the original.

    Args:
        in_channels (int): Number of input channels (bands * electrode_channels)
        bottleneck_channels (int): Number of channels in the bottleneck representation
        freq_dim (int): Dimension of the frequency bins
    """

    def __init__(
        self,
        in_channels: int = 32,  # 2 bands * 16 electrode channels
        bottleneck_channels: int = 16,
        freq_dim: int = 0,  # Will be inferred from input if 0
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.bottleneck_channels = bottleneck_channels
        self.freq_dim = freq_dim

        # Encoder: compress from in_channels to bottleneck_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, bottleneck_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(),
        )

        # Decoder: reconstruct from bottleneck_channels to in_channels
        self.decoder = nn.Sequential(
            nn.Conv2d(bottleneck_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs: Input tensor of shape (T, N, bands, electrode_channels, freq)
                  or (N, bands * electrode_channels, freq, T)

        Returns:
            tuple containing:
                - encoded: Bottleneck representation (T, N, bottleneck_channels, freq)
                - reconstructed: Reconstructed input (T, N, bands, electrode_channels, freq)
        """
        # Determine if we need to reshape
        if len(inputs.shape) == 5:  # (T, N, bands, electrode_channels, freq)
            T, N, bands, C, freq = inputs.shape
            # Reshape to (N, bands*C, freq, T) for 2D convolutions
            x = inputs.permute(1, 2, 3, 4, 0).reshape(N, bands * C, freq, T)
            original_shape = inputs.shape
        elif len(inputs.shape) == 4:  # (N, bands*C, freq, T)
            N, C, freq, T = inputs.shape
            x = inputs
            # Save shape for reshaping back later
            original_shape = (T, N, 2, C // 2, freq)  # Assuming 2 bands
        else:
            raise ValueError(f"Unexpected input shape: {inputs.shape}")

        # Update freq_dim if needed
        if self.freq_dim == 0:
            self.freq_dim = freq

        # Encode
        encoded: torch.Tensor = self.encoder(x)

        # Decode
        reconstructed: torch.Tensor = self.decoder(encoded)

        # Reshape encoded to (T, N, bottleneck_channels, freq)
        encoded_reshaped = encoded.reshape(N, self.bottleneck_channels, freq, T).permute(3, 0, 1, 2)

        # Reshape reconstructed back to original shape if needed
        if len(inputs.shape) == 5:
            reconstructed = reconstructed.reshape(N, bands, C, freq, T).permute(4, 0, 1, 2, 3)
        else:
            reconstructed = reconstructed.reshape(
                N, original_shape[2], original_shape[3], freq, T
            ).permute(4, 0, 1, 2, 3)

        return encoded_reshaped, reconstructed

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Get only the encoded representation.

        Args:
            inputs: Input tensor

        Returns:
            encoded: Bottleneck representation
        """
        encoded, _ = self.forward(inputs)
        return encoded

    def get_reconstruction_loss(self, inputs: torch.Tensor) -> torch.Tensor:
        """Calculate reconstruction loss for training the autoencoder.

        Args:
            inputs: Input tensor

        Returns:
            loss: MSE reconstruction loss
        """
        _, reconstructed = self.forward(inputs)
        return nn.functional.mse_loss(reconstructed, inputs)
