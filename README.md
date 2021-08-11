# emg2qwerty

Code accompanying `emg2qwerty` dataset along with baselines.

### Setup

```shell
# Make sure you have [git-lfs installed](https://git-lfs.github.com/) (for pre-trained models)
git lfs install

git clone git@github.com:fairinternal/emg2qwerty.git && cd emg2qwerty/
conda env create -f environment.yml

ln -s /private/home/viswanath/oss/emg2qwerty-data-full-2021-07-30 data
```

### Training on FAIR Cluster

Generic user model:
```shell
python -m emg2qwerty.train \
  user=generic \
  trainer.gpus=8 \
  +cluster=slurm -m
```

Personalized user models:
```shell
python -m emg2qwerty.train \
  user="glob(user*)" \
  trainer.gpus=1 \
  checkpoint="${HOME}/emg2qwerty/models/generic.ckpt" \
  +cluster=slurm -m

```

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
  +cluster=slurm -m
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
  +cluster=slurm -m
```

The above uses a 6-gram character-level language model trained using kenlm on WikiText-103 raw corpus. To re-build the language model,
1. Build kenlm from source: https://github.com/kpu/kenlm#compiling
2. Download nltk punkt tokenizer: `python -c "import nltk; nltk.download('punkt')"`
3. Run `./scripts/lm/build_char_lm.sh <ngram_order>`
