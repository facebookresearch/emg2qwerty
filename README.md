# emg2qwerty

A dataset of Surface electromyography (sEMG) recordings while typing on a QWERTY keyboard with ground-truth, benchmarks and baselines.

<p align="center">
  <img src="https://user-images.githubusercontent.com/172884/131009650-a8e96814-7ed6-4ba2-b995-a9f7a39c858c.png" alt="EMG segment" height="80%" width="80%">
</p>

### Setup

**Update:** As of July 2022, the URL below does not work. Follow the instructions in https://www.internalfb.com/intern/wiki/RL/RL-CTRL/FRL-CTRL_Engineering_Overview/RIS/CTRL-FAIR_Collaboration_Guide/ to access this dataset on the FAIR cluster.

```shell
# Download the dataset
wget https://fb-ctrl-oss.s3.amazonaws.com/emg2qwerty/emg2qwerty-data-2021-08.tar.gz
tar -xvzf emg2qwerty-data-2021-08.tar.gz

# Make sure you have [git-lfs installed](https://git-lfs.github.com/) (for pre-trained models)
git lfs install

git clone git@github.com:fairinternal/emg2qwerty.git && cd emg2qwerty/
conda env create -f environment.yml

# Symlink the dataset directory
ln -s ~/emg2qwerty-data-2021-08 data
```

### Data

The dataset consists of 1,136 files in total - 1,135 session files spanning 108 users and 346 hours of recording, and one `metadata.csv` file. Each session file is in a simple HDF5 format and includes the left and right EMG signal data, prompted text, keylogger ground-truth, and their corresponding timestamps. `emg2qwerty.data.Emg2QwertySessionData` offers a programmatic read-only interface into the HDF5 session files.

To load the `metadata.csv` file and print dataset statistics,
```shell
python scripts/print_dataset_stats.py
```

<p align="center">
  <img src="https://user-images.githubusercontent.com/172884/131012947-66cab4c4-963c-4f1a-af12-47fea1681f09.png" alt="Dataset statistics" height="50%" width="50%">
</p>

To re-generate data splits,
```shell
python scripts/generate_splits.py
```

The following figure visualizes the dataset splits for training, validation and testing of generic and personalized user models. Refer to the paper for details of the benchmark setup and data splits.

<p align="center">
  <img src="https://user-images.githubusercontent.com/172884/131012465-504eccbf-8eac-4432-b8aa-0e453ad85b49.png" alt="Data splits">
</p>

To re-format data in [EEG BIDS format](https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/03-electroencephalography.html),
```shell
python scripts/convert_to_bids.py
```


### Training

Generic user model:
```shell
python -m emg2qwerty.train user=generic trainer.gpus=8 +cluster=local -m
```

Personalized user models:
```shell
python -m emg2qwerty.train \
  user="glob(user*)" \
  trainer.gpus=1 \
  checkpoint="${HOME}/emg2qwerty/models/generic.ckpt" \
  +cluster=local -m

```

If you are using a Slurm cluster, set "+cluster=slurm" in the above commands to pick up `config/cluster/slurm.yaml`. This overrides the Hydra Launcher to use [Submitit plugin](https://hydra.cc/docs/plugins/submitit_launcher). Refer to Hydra documentation for the list of available launcher plugins if you are not using a Slurm cluster.

### Testing

Greedy decoding:
```shell
python -m emg2qwerty.train \
  user=user0 \
  decoder=ctc_greedy \
  train=False \
  trainer.gpus=0 \
  checkpoint="${HOME}/emg2qwerty/models/user0.ckpt" \
  hydra.launcher.mem_gb=64 \
  +cluster=local -m
```

Beam-search decoding with 6-gram language model:
```shell
python -m emg2qwerty.train \
  user=user0 \
  decoder=ctc_beam \
  train=False \
  trainer.gpus=0 \
  checkpoint="${HOME}/emg2qwerty/models/user0.ckpt" \
  hydra.launcher.mem_gb=64 \
  +cluster=local -m
```

The above uses a 6-gram character-level language model trained using kenlm on WikiText-103 raw corpus. To re-build the language model,
1. Build kenlm from source: https://github.com/kpu/kenlm#compiling
2. Download nltk punkt tokenizer: `python -c "import nltk; nltk.download('punkt')"`
3. Run `./scripts/lm/build_char_lm.sh <ngram_order>`
