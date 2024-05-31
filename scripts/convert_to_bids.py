# Copyright (c) Meta Platforms, Inc. and its affiliates.
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
import yaml

from emg2qwerty.data import EMGSessionData

mne.set_log_level("WARNING")


def get_raw(session_name: str, dataset_root: str) -> mne.io.Raw:
    """Read data in HDF5 and get an MNE Raw object with keystrokes and prompts
    saved as mne.Annotations."""
    fname = Path(dataset_root) / (session_name + ".hdf5")
    session = EMGSessionData(fname)
    label_data = session.ground_truth()

    ch_names = [f"emg{i}" for i in range(16)]
    ch_names = [f"{ch}_left" for ch in ch_names] + [f"{ch}_right" for ch in ch_names]
    sfreq = 2000.0  # Hz
    data = np.concatenate((session["emg_left"], session["emg_right"]), axis=1)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data.T, info)

    # Deal with annot for keystrokes
    idx = np.searchsorted(session["time"], label_data.timestamps)
    # Fix idx as certain keys arrive after then final timestamp
    idx[idx >= len(session["time"])] = len(session["time"]) - 1

    chars = list(label_data.text)
    mapping = {
        "⌫": "Key.backspace",
        "⏎": "Key.enter",
        "⇧": "Key.shift",
        " ": "Key.space",
    }
    chars_ = [mapping.get(c, c) for c in chars]
    # prefix with "key" to distinguish with prompts
    chars_ = ["key/" + c for c in chars_]
    annot = mne.Annotations(
        onset=raw.times[idx], duration=np.zeros(len(idx)), description=chars_
    )

    # Deal with annot for prompts
    prompts = pd.DataFrame(session.prompts)
    prompts = prompts.query("name == 'text_prompt'")
    idx_start = np.searchsorted(session["time"], prompts.start)
    idx_end = np.searchsorted(session["time"], prompts.end)
    # Fix idx as certain keys arrive after then final timestamp
    idx_start[idx_start >= len(session["time"])] = len(session["time"]) - 1
    idx_end[idx_end >= len(session["time"])] = len(session["time"]) - 1
    onset = raw.times[idx_start]
    duration = raw.times[idx_end] - raw.times[idx_start]
    description = prompts.payload.apply(pd.Series).text.str.replace("⏎", "\\n").values
    # prefix with "prompt" to distinguish with keys
    description = ["prompt/" + d for d in description]
    annot_prompts = mne.Annotations(
        onset=onset, duration=duration, description=description
    )

    annot = annot + annot_prompts
    raw.set_annotations(annot)
    return raw


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
def main(
    dataset_root: str,
    bids_root: str,
):
    df = pd.read_csv(Path(dataset_root).joinpath("metadata.csv"))
    df.quality_check_tags = df.quality_check_tags.apply(yaml.safe_load)

    for ix_subject, (_, df_user) in enumerate(df.groupby("user")):
        for ix_session in tqdm.tqdm(range(len(df_user.session.values))):
            raw = get_raw(df_user.iloc[ix_session].session, dataset_root)
            bids_path = mne_bids.BIDSPath(
                subject=f"{ix_subject + 1:02d}",
                session=f"{ix_session + 1:02d}",
                task="typing",
                root=bids_root,
            )
            mne_bids.write_raw_bids(
                raw, bids_path, overwrite=True, format="BrainVision", allow_preload=True
            )


if __name__ == "__main__":
    main()
