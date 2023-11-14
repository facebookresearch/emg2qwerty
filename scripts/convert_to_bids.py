"""Script to convert dataset to BIDS format (pretending it's EEG)"""
# %%

from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import mne
import mne_bids
import tqdm

mne.set_log_level("WARNING")

# %%

dataset_root = Path("./data")
df = pd.read_csv(dataset_root.joinpath("metadata.csv"))
df.quality_check_tags = df.quality_check_tags.apply(yaml.safe_load)
df.head()

# %%
from emg2qwerty.data import Emg2QwertySessionData


def get_raw(session_name):
    fname = dataset_root / (session_name + ".hdf5")
    session = Emg2QwertySessionData(fname)
    # 1 / np.median(np.diff(session["time"]))

    label_data = session.ground_truth()
    # label_data.labels
    # label_data.label_str
    # label_data.timestamps.shape

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

    chars = list(label_data.label_str)
    mapping = {
        "⌫": "Key.backspace",
        "⏎": "Key.enter",
        "⇧": "Key.shift",
        " ": "Key.space",
    }
    chars_ = [mapping[c] if c in mapping else c for c in chars]
    chars_ = ["key/" + c for c in chars_]  # prefix with "key" to distinguish with prompts
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
    description = ["prompt/" + d for d in description]  # prefix with "prompt" to distinguish with keys
    annot_prompts = mne.Annotations(
        onset=onset, duration=duration, description=description
    )

    annot = annot + annot_prompts
    raw.set_annotations(annot)
    return raw


# raw = get_raw(df.session.iloc[0])
# raw.save("tmp_with_annot_raw.fif", overwrite=True)

# %%

for ix_subject, (user, df_user) in enumerate(df.groupby("user")):
    for ix_session in tqdm.tqdm(range(len(df_user.session.values))):
        raw = get_raw(df_user.iloc[ix_session].session)
        bids_path = mne_bids.BIDSPath(
            subject=f"{ix_subject + 1:02d}",
            session=f"{ix_session + 1:02d}",
            task="typing",
            root="bids_data",
        )
        mne_bids.write_raw_bids(
            raw, bids_path, overwrite=True, format="BrainVision", allow_preload=True
        )
