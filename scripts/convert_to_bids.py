# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Script to convert the dataset from HDF5 to BIDS format (pretending it's EEG).
https://bids-specification.readthedocs.io/en/stable/modality-specific-files/electroencephalography.html.
"""

from pathlib import Path

import click
import mne
import mne_bids
import numpy as np
import pandas as pd
import tqdm
from emg2qwerty.charset import charset

from emg2qwerty.data import EMGSessionData

mne.set_log_level("WARNING")


def get_mne_raw(session_path: Path) -> mne.io.Raw:
    """Read data in HDF5 and get an MNE Raw object with keystrokes and prompts
    saved as mne.Annotations."""
    session = EMGSessionData(session_path)
    label_data = session.ground_truth()

    ch_names = [f"emg{i}" for i in range(16)]
    ch_names = [f"{ch}_left" for ch in ch_names] + [f"{ch}_right" for ch in ch_names]
    sfreq = 2000.0  # Hz
    data = np.concatenate(
        (session[EMGSessionData.EMG_LEFT], session[EMGSessionData.EMG_RIGHT]), axis=1
    )
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data.T, info)

    # Deal with annotations for keystrokes
    timestamps = session.timestamps
    idx = np.searchsorted(timestamps, label_data.timestamps)
    # Fix idx as certain keys arrive after the final timestamp
    idx[idx >= len(timestamps)] = len(timestamps) - 1

    keys = charset().str_to_keys(label_data.text)
    # Prefix with "key" to distinguish from prompts
    keys = [f"key/{key}" for key in keys]
    annotation_keys = mne.Annotations(
        onset=raw.times[idx],
        duration=np.zeros(len(idx)),
        description=keys,
    )

    # Deal with annotations for prompts
    prompts = pd.DataFrame(session.prompts)
    prompts = prompts.query("name == 'text_prompt'")
    idx_start = np.searchsorted(timestamps, prompts.start)
    idx_end = np.searchsorted(timestamps, prompts.end)
    # Fix idx as certain keys arrive after the final timestamp
    idx_start[idx_start >= len(timestamps)] = len(timestamps) - 1
    idx_end[idx_end >= len(timestamps)] = len(timestamps) - 1
    onset = raw.times[idx_start]
    duration = raw.times[idx_end] - raw.times[idx_start]
    description = prompts.payload.apply(pd.Series).text.str.replace("⏎", "\\n").values
    # Prefix with "prompt" to distinguish from keys
    description = ["prompt/" + d for d in description]
    annotation_prompts = mne.Annotations(
        onset=onset,
        duration=duration,
        description=description,
    )

    annotations = annotation_keys + annotation_prompts
    raw.set_annotations(annotations)
    return raw


def convert_to_bids(
    subject_idx: int,
    session_idx: int,
    session_path: Path,
    bids_root: str,
) -> None:
    """Convert the HDF5 emg2qwerty session file located at `session_path` to BIDS format
    and output to `bids_root` directory.
    """
    raw = get_mne_raw(session_path)
    bids_path = mne_bids.BIDSPath(
        subject=f"{subject_idx + 1:02d}",
        session=f"{session_idx + 1:02d}",
        task="emg2qwerty",
        root=bids_root,
    )
    mne_bids.write_raw_bids(
        raw=raw,
        bids_path=bids_path,
        overwrite=True,
        format="BrainVision",
        allow_preload=True,
    )


@click.command()
@click.option(
    "--dataset-root",
    type=str,
    default=Path(__file__).parents[1].joinpath("data"),
    help="Original dataset root directory (defaults to 'data' folder)",
)
@click.option(
    "--bids-root",
    type=str,
    default=Path(__file__).parents[1].joinpath("bids_data"),
    help="BIDS dataset root directory (defaults to 'bids_data' folder)",
)
def main(dataset_root: str, bids_root: str):
    df = pd.read_csv(Path(dataset_root).joinpath("metadata.csv"))
    users = sorted(df["user"].unique())
    for subject_idx, user in enumerate(users):
        sessions = sorted(df[df["user"] == user].session)
        for session_idx, session in enumerate(tqdm.tqdm(sessions)):
            session_path = Path(dataset_root).joinpath(f"{session}.hdf5")
            convert_to_bids(
                subject_idx=subject_idx,
                session_idx=session_idx,
                session_path=session_path,
                bids_root=bids_root,
            )


if __name__ == "__main__":
    main()
