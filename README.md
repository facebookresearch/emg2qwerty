# emg2qwerty

[ [`Paper`](https://arxiv.org/abs/2410.20081) ] [ [`Dataset`](https://fb-ctrl-oss.s3.amazonaws.com/emg2qwerty/emg2qwerty-data-2021-08.tar.gz) ] [ [`Blog`](https://ai.meta.com/blog/open-sourcing-surface-electromyography-datasets-neurips-2024/) ] [ [`BibTeX`](#citing-emg2qwerty) ]

A dataset of surface electromyography (sEMG) recordings while touch typing on a QWERTY keyboard with ground-truth, benchmarks and baselines.

<p align="center">
  <img src="https://github.com/user-attachments/assets/71a9f361-7685-4188-83c3-099a009b6b81" height="80%" width="80%" alt="alt="sEMG recording" >
</p>

## Setup

```shell
# Install [git-lfs](https://git-lfs.github.com/) (for pretrained checkpoints)
git lfs install

# Clone the repo, setup environment, and install local package
git clone git@github.com:facebookresearch/emg2qwerty.git ~/emg2qwerty
cd ~/emg2qwerty
conda env create -f environment.yml
conda activate emg2qwerty
pip install -e .

# Download the dataset, extract, and symlink to ~/emg2qwerty/data
cd ~ && wget https://fb-ctrl-oss.s3.amazonaws.com/emg2qwerty/emg2qwerty-data-2021-08.tar.gz
tar -xvzf emg2qwerty-data-2021-08.tar.gz
ln -s ~/emg2qwerty-data-2021-08 ~/emg2qwerty/data
```

## Data

The dataset consists of 1,136 files in total - 1,135 session files spanning 108 users and 346 hours of recording, and one `metadata.csv` file. Each session file is in a simple HDF5 format and includes the left and right sEMG signal data, prompted text, keylogger ground-truth, and their corresponding timestamps. `emg2qwerty.data.EMGSessionData` offers a programmatic read-only interface into the HDF5 session files.

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

## Training

Generic user model:

```shell
python -m emg2qwerty.train \
  user=generic \
  trainer.accelerator=gpu trainer.devices=8 \
  --multirun
```

Personalized user models:

```shell
python -m emg2qwerty.train \
  user="glob(user*)" \
  trainer.accelerator=gpu trainer.devices=1 \
  checkpoint="${HOME}/emg2qwerty/models/generic.ckpt" \
  --multirun
```

Personalized config:

```shell
python -m emg2qwerty.train --config-name [config] [overrides]
```

If you are using a Slurm cluster, include "cluster=slurm" override in the argument list of above commands to pick up `config/cluster/slurm.yaml`. This overrides the Hydra Launcher to use [Submitit plugin](https://hydra.cc/docs/plugins/submitit_launcher). Refer to Hydra documentation for the list of available launcher plugins if you are not using a Slurm cluster.

## Testing

Greedy decoding:

```shell
python -m emg2qwerty.train \
  user="glob(user*)" \
  checkpoint="${HOME}/emg2qwerty/models/personalized-finetuned/\${user}.ckpt" \
  train=False trainer.accelerator=cpu \
  decoder=ctc_greedy \
  hydra.launcher.mem_gb=64 \
  --multirun
```

Beam-search decoding with 6-gram character-level language model:

```shell
python -m emg2qwerty.train \
  user="glob(user*)" \
  checkpoint="${HOME}/emg2qwerty/models/personalized-finetuned/\${user}.ckpt" \
  train=False trainer.accelerator=cpu \
  decoder=ctc_beam \
  hydra.launcher.mem_gb=64 \
  --multirun
```

The 6-gram character-level language model, used by the first-pass beam-search decoder above, is generated from [WikiText-103 raw dataset](https://huggingface.co/datasets/wikitext), and built using [KenLM](https://github.com/kpu/kenlm). The LM is available under `models/lm/`, both in the binary format, and the human-readable [ARPA format](https://cmusphinx.github.io/wiki/arpaformat/). These can be regenerated as follows:

1. Build kenlm from source: <https://github.com/kpu/kenlm#compiling>
2. Run `./scripts/lm/build_char_lm.sh <ngram_order>`

## License

emg2qwerty is CC-BY-NC-4.0 licensed, as found in the LICENSE file.

## Citing emg2qwerty

```
@misc{sivakumar2024emg2qwertylargedatasetbaselines,
      title={emg2qwerty: A Large Dataset with Baselines for Touch Typing using Surface Electromyography},
      author={Viswanath Sivakumar and Jeffrey Seely and Alan Du and Sean R Bittner and Adam Berenzweig and Anuoluwapo Bolarinwa and Alexandre Gramfort and Michael I Mandel},
      year={2024},
      eprint={2410.20081},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.20081},
}
```
