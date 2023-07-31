# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Dict,
    KeysView,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import h5py
import numpy as np
import torch
from torch import nn

from emg2qwerty import transforms
from emg2qwerty.charset import CharacterSet, charset
from emg2qwerty.transforms import Transform


@dataclass
class Emg2QwertySessionData:
    """A read-only interface to a single emg2qwerty session file stored in
    HDF5 format.

    A session here refers to a span of a few minutes during which two-handed
    EMG signals were recorded while a user typed out a series of prompted
    words/sentences. This class encapsulates the EMG timeseries, ground-truth,
    and additional metadata corresponding to a single session.

    ``self.timeseries`` is a `h5py.Dataset` instance with a compound data type
    as in a numpy structured array containing three fields - EMG data from the
    left and right wrists, and their corresponding timestamps.
    The sampling rate of EMG is 2kHz, each EMG device has 16 electrode
    channels, and the signal has been high-pass filtered. Therefore, the fields
    corresponding to left and right EMG are 2D arrays of shape ``(T, 16)`` each
    and ``timestamps`` is a 1D array of length ``T``.

    ``self.metadata`` contains two kinds of ground-truth:
      1. A sequence of ``prompts`` displayed to the user (where each prompt
         is a handful of words) along with their start and end timestamps.
         This offers less accurate ground-truth ground-truth as there is
         no guarantee that the user actually typed the prompted words
         accurately without typos. It also lacks time alignment of each
         key-press within the prompt window.
      2. A sequence of ``keystrokes`` indicating the key-presses on a keyboard
         as recorded by keylogger along with the timestamps corresponding to
         individual key-presses and the key-releases. This offers high-quality
         and accurate ground-truth as well as temporal alignment of EMG window
         with each key character.

    NOTE: Only the metadata and ground-truth are loaded into memory while the
    EMG data is accesssed directly from disk. When wrapping this interface
    within a PyTorch Dataset, use multiple dataloading workers to mask the
    disk seek and read latencies."""

    HDF5_GROUP: ClassVar[str] = "emg2qwerty"
    TIMESERIES: ClassVar[str] = "timeseries"
    EMG_LEFT: ClassVar[str] = "emg_left"
    EMG_RIGHT: ClassVar[str] = "emg_right"
    TIMESTAMPS: ClassVar[str] = "time"
    SESSION_NAME: ClassVar[str] = "session_name"
    USER: ClassVar[str] = "user"
    CONDITION: ClassVar[str] = "condition"
    DURATION_MINS: ClassVar[str] = "duration_mins"
    KEYSTROKES: ClassVar[str] = "keystrokes"
    PROMPTS: ClassVar[str] = "prompts"

    hdf5_path: Path

    def __post_init__(self) -> None:
        self._file = h5py.File(self.hdf5_path, "r")
        emg2qwerty_group: h5py.Group = self._file[self.HDF5_GROUP]

        # ``timeseries`` is a HDF5 compound Dataset with aligned left and
        # right EMG data along with corresponding timestamps. Don't load
        # the data into memory - users should instead rely on PyTorch
        # DataLoader workers to mask the disk seek/read latency.
        self.timeseries: h5py.Dataset = emg2qwerty_group[self.TIMESERIES]
        assert self.timeseries.dtype.fields is not None
        assert self.EMG_LEFT in self.timeseries.dtype.fields
        assert self.EMG_RIGHT in self.timeseries.dtype.fields
        assert self.TIMESTAMPS in self.timeseries.dtype.fields

        # Load the metadata entirely into memory as it's rather small
        self.metadata: Dict[str, Any] = {}
        for key, val in emg2qwerty_group.attrs.items():
            if key in [self.KEYSTROKES, self.PROMPTS]:
                self.metadata[key] = json.loads(val)
            else:
                self.metadata[key] = val

    def __enter__(self) -> "Emg2QwertySessionData":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._file.close()

    def __len__(self) -> int:
        return len(self.timeseries)

    def __getitem__(self, key: slice) -> np.ndarray:
        return self.timeseries[key]

    def slice(self, start_t: float = -np.inf, end_t: float = np.inf) -> np.ndarray:
        """Load and return a contiguous slice of the timeseries windowed
        by the provided start and end timestamps.

        Args:
            start_t (float): The start time of the window to grab
                (in absolute unix time). Defaults to selecting from the
                beginning of the session. (default: ``-np.inf``).
            end_t (float): The end time of the window to grab
                (in absolute unix time). Defaults to selecting until the
                end of the session. (default: ``np.inf``)
        """
        start_idx, end_idx = self.timestamps.searchsorted([start_t, end_t])
        return self[start_idx:end_idx]

    def ground_truth(
        self, start_t: float = -np.inf, end_t: float = np.inf
    ) -> "LabelData":
        if self.condition == "on_keyboard":
            return LabelData.from_keystrokes(
                self.keystrokes, start_t=start_t, end_t=end_t
            )
        else:
            return LabelData.from_prompts(self.prompts, start_t=start_t, end_t=end_t)

    @property
    def fields(self) -> KeysView[str]:
        """The names of the fields in ``timeseries``."""
        fields: KeysView[str] = self.timeseries.dtype.fields.keys()
        return fields

    @property
    def timestamps(self) -> np.ndarray:
        """EMG timestamps.

        NOTE: This reads the entire sequence of timesetamps from the underlying
        HDF5 file and therefore incurs disk latency. Avoid this in the critical
        path."""
        emg_timestamps = self.timeseries[self.TIMESTAMPS]
        assert (np.diff(emg_timestamps) >= 0).all(), "Not monotonic"
        return emg_timestamps

    @property
    def session_name(self) -> str:
        """Unique name of the session."""
        return self.metadata[self.SESSION_NAME]

    @property
    def user(self) -> str:
        """Unique ID of the user this session corresponds to."""
        return self.metadata[self.USER]

    @property
    def condition(self) -> str:
        return self.metadata[self.CONDITION]

    @property
    def duration_mins(self) -> float:
        """The duration of the EMG session in minutes."""
        return self.metadata[self.DURATION_MINS]

    @property
    def keystrokes(self) -> List[Dict[str, Any]]:
        """Sequence of keys recorded by the keylogger during the
        data-collection session along with the press and release timestamps
        for each key."""
        return self.metadata[self.KEYSTROKES]

    @property
    def prompts(self) -> List[Dict[str, Any]]:
        """Sequence of sentences prompted to the user during the
        data-collection session along with the start and end timestamps
        for each prompt."""
        return self.metadata[self.PROMPTS]

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__} {self.session_name} " f"({len(self)} samples)"
        )


@dataclass
class LabelData:
    """Canonical representation for label data together with optional
    character-level timestamps. Supports standarization from
    keylogger keystrokes, prompts, and pynput key representations.

    NOTE: Avoid calling ``LabelData`` constructor directly and instead
    use the provided factory classmethods as much as possible."""

    _charset: ClassVar[CharacterSet] = charset()

    label_str: str
    _timestamps: InitVar[Optional[Sequence[float]]] = None

    def __post_init__(self, _timestamps: Optional[Sequence[float]]) -> None:
        self.timestamps: Optional[np.ndarray] = None
        if _timestamps is not None:
            self.timestamps = np.array(_timestamps)
            assert self.timestamps.ndim == 1
            assert len(self.timestamps) == len(self.label_str)
            assert (np.diff(self.timestamps) >= 0).all(), "Not monotonic"

    @classmethod
    def from_keystrokes(
        cls,
        keystrokes: Sequence[Mapping[str, Any]],
        start_t: float = -np.inf,
        end_t: float = np.inf,
    ) -> "LabelData":
        """Create a new instance of ``LabelData`` from a sequence of
        keystrokes between the provided start and end timestamps, after
        normalizing and cleaning up as per ``CharacterSet``. The returned
        object also includes the key-press timestamps corresponding to each
        character in ``self.label_str``.

        Args:
            keystrokes (list): Each keystroke entry in the list should be a
                dict in the format of ``Emg2QwertySessionData.keystrokes``.
            start_t (float): The start timestamp of the window in absolute
                unix time. (default: ``-np.inf``)
            end_t (float): The end timestamp of the window in absolute
                unix time. (default: ``np.inf``)
        """
        label_data = cls(label_str="", _timestamps=[])
        for keystroke in keystrokes:
            if keystroke["start"] > end_t:
                break
            if keystroke["start"] >= start_t:
                label_data += cls.from_keystroke(keystroke)
        return label_data

    @classmethod
    def from_keystroke(
        cls, keystroke: Union[str, Mapping[str, Any]], timestamp: Optional[float] = None
    ) -> "LabelData":
        """Create a new instance of ``LabelData`` from a single keystroke,
        after normalizing and cleaning up as per ``CharacterSet``.

        Args:
            keystroke (str or dict): A single pynput.Key string or a keystroke
                dict in the format of ``Emg2QwertySessionData.keystrokes``.
            timestamp (float): Optional timestamp corresponding to the
                keystroke. If not provided and if ``keystroke`` is a dict,
                this will be set to the key-press time available in the dict.
                (default: ``None``)
        """
        if isinstance(keystroke, str):
            key = keystroke
        else:
            key = keystroke["key"]
            timestamp = keystroke["start"] if timestamp is None else timestamp

        key = cls._charset._normalize_keys([key])[0]
        if key not in cls._charset:  # Out of vocabulary
            return cls(label_str="", _timestamps=[])

        label_str = cls._charset.keys_to_str([key])
        timestamps = [timestamp] if timestamp is not None else None
        return cls(label_str, timestamps)

    @classmethod
    def from_prompts(
        cls,
        prompts: Sequence[Mapping[str, Any]],
        enforce_newline: bool = True,
        start_t: float = -np.inf,
        end_t: float = np.inf,
    ) -> "LabelData":
        """Create a new instance of ``LabelData`` from a sequence of prompts
        between the provided start and end timestamps, after normalizing and
        cleaning up as per ``CharacterSet``. The returned object does not
        include character-level timestamps.

        Args:
            prompts (list): Each prompt entry in the list should be a dict in
                the format of ``Emg2QwertySessionData.prompts``.
            enforce_newline (bool): If set, end each prompt with a newline
                if not present already. (default: ``True``)
            start_t (float): The start timestamp of the window in absolute
                unix time. (default: ``-np.inf``)
            end_t (float): The end timestamp of the window in absolute
                unix time. (default: ``np.inf``)
        """
        label_data = cls(label_str="")
        for prompt in prompts:
            if prompt["start"] > end_t:
                break
            if prompt["start"] >= start_t:
                label_data += cls.from_prompt(prompt, enforce_newline=enforce_newline)
        return label_data

    @classmethod
    def from_prompt(
        cls, prompt: Union[str, Mapping[str, Any]], enforce_newline: bool = True
    ) -> "LabelData":
        """Create a new instance of ``LabelData`` from a single prompt, after
        normalizing and cleaning up as per ``CharacterSet``. The returned
        object does not include character-level timestamps.

        Args:
            prompt (str or dict): A single prompt, either as raw text or a
                dict in the format of ``Emg2QwertySessionData.prompts``.
            enforce_newline (bool): If set, end the prompt with a newline
                if not present already. (default: ``True``)
        """
        if isinstance(prompt, str):
            label_str = prompt
        else:
            payload = prompt["payload"]
            label_str = payload["text"] if payload is not None else None

        # Do not add terminal newline if there was no prompt payload
        if label_str is None:
            return cls(label_str="")

        label_str = cls._charset.clean_str(label_str)
        if enforce_newline and (len(label_str) == 0 or label_str[-1] != "⏎"):
            label_str += "⏎"
        return cls(label_str)

    @classmethod
    def from_labels(
        cls, labels: Sequence[int], timestamps: Optional[Sequence[float]] = None
    ) -> "LabelData":
        """Create a new instance of ``LabelData`` from integer labels
        and optionally together with its corresponding timestamps."""
        label_str = cls._charset.labels_to_str(labels)
        return cls(label_str, timestamps)

    @property
    def labels(self) -> np.ndarray:
        """Integer labels corresponding to the label string."""
        labels = LabelData._charset.str_to_labels(self.label_str)
        return np.asarray(labels, dtype=np.int32)

    def __len__(self) -> int:
        return len(self.label_str)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LabelData):
            return NotImplemented
        return self.label_str == other.label_str

    def __add__(self, other: "LabelData") -> "LabelData":
        if not isinstance(other, LabelData):
            return NotImplemented

        label_str = self.label_str + other.label_str
        if self.timestamps is not None and other.timestamps is not None:
            timestamps = np.append(self.timestamps, other.timestamps)
        else:
            timestamps = None

        return LabelData(label_str, timestamps)

    def __str__(self) -> str:
        """Human-readable string representation for display."""
        return self.label_str.replace("⏎", "\n")


@dataclass
class WindowedEmgDataset(torch.utils.data.Dataset):
    """A `torch.utils.data.Dataset` corresponding to an instance of
    `Emg2QwertySessionData` that iterates over EMG windows of configurable
    length and stride.

    Args:
        hdf5_path (str): Path to the session file in hdf5 format.
        window_length (int): Size of each window. Specify None for no windowing
            in which case this will be a dataset of length 1 containing the
            entire session. (default: ``None``)
        stride (int): Stride between consecutive windows. Specify None to set
            this to window_length, in which case there will be no overlap
            between consecutive windows. (default: ``window_length``)
        padding (Tuple[int, int]): Left and right contextual padding for
            windows in terms of number of raw EMG samples.
        jitter (bool): If True, randomly jitter the offset of each window.
            Use this for training time variability. (default: ``False``)
        transform (Callable): A composed sequence of transforms that takes
            a window/slice of `Emg2QwertySessionData` in the form of a numpy
            structured array and returns a `torch.Tensor` instance.
            (default: ``emg2qwerty.transforms.ToTensor()``)
    """

    hdf5_path: Path
    window_length: InitVar[Optional[int]] = None
    stride: InitVar[Optional[int]] = None
    padding: InitVar[Tuple[int, int]] = (0, 0)
    jitter: bool = False
    transform: Transform[np.ndarray, torch.Tensor] = field(
        default_factory=lambda: transforms.ToTensor()
    )

    def __post_init__(
        self,
        window_length: Optional[int],
        stride: Optional[int],
        padding: Tuple[int, int],
    ) -> None:
        with Emg2QwertySessionData(self.hdf5_path) as session:
            assert (
                session.condition == "on_keyboard"
            ), f"Unsupported condition {self.session.condition}"
            self.session_length = len(session)

        self.window_length = (
            window_length if window_length is not None else self.session_length
        )
        self.stride = stride if stride is not None else self.window_length
        assert self.window_length > 0 and self.stride > 0

        (self.left_padding, self.right_padding) = padding
        assert self.left_padding >= 0 and self.right_padding >= 0

    def __len__(self) -> int:
        return (self.session_length - self.window_length) // self.stride + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Lazy init `Emg2QwertySessionData` per dataloading worker
        # since `h5py.File` objects can't be picked
        if not hasattr(self, "session"):
            self.session = Emg2QwertySessionData(self.hdf5_path)

        offset = idx * self.stride

        # Randomly jitter the window offset
        leftover = len(self.session) - (offset + self.window_length)
        if leftover < 0:
            raise IndexError(f"Index {idx} out of bounds")
        if leftover > 0 and self.jitter:
            offset += np.random.randint(0, min(self.stride, leftover))

        # Expand window to include contextual padding and fetch
        window_start = max(offset - self.left_padding, 0)
        window_end = offset + self.window_length + self.right_padding
        window = self.session[window_start:window_end]

        # Extract EMG tensor corresponding to the window
        emg = self.transform(window)
        assert torch.is_tensor(emg)

        # Extract labels corresponding to the original (un-padded) window
        timestamps = window[Emg2QwertySessionData.TIMESTAMPS]
        start_t = timestamps[offset - window_start]
        end_t = timestamps[(offset + self.window_length - 1) - window_start]
        label_data = self.session.ground_truth(start_t, end_t)
        labels = torch.as_tensor(label_data.labels)

        return emg, labels

    @staticmethod
    def collate(
        samples: Sequence[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        inputs = [sample[0] for sample in samples]  # [(T, ...)]
        targets = [sample[1] for sample in samples]  # [(T,)]

        # Batch of inputs and targets padded along time
        input_batch = nn.utils.rnn.pad_sequence(inputs)  # (T, N, ...)
        target_batch = nn.utils.rnn.pad_sequence(targets)  # (T, N)

        # Lengths of unpadded input and target sequences for each batch entry
        input_lengths = torch.as_tensor(
            [len(input) for input in inputs], dtype=torch.int32
        )
        target_lengths = torch.as_tensor(
            [len(target) for target in targets], dtype=torch.int32
        )

        return {
            "inputs": input_batch,
            "targets": target_batch,
            "input_lengths": input_lengths,
            "target_lengths": target_lengths,
        }
