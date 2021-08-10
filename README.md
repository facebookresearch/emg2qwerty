# emg2qwerty

Code accompanying `emg2qwerty` dataset along with baselines.

### Setup

```shell
conda env create -f environment.yml
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
