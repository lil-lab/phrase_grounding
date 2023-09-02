# Touchdown SDR Experiments Guide

Welcome to the Touchdown SDR Experiments Guide. Follow the steps below to fine-tune and test models using the ViLT-Aligner.

## Using the ViLT-Aligner

### 1. Downloading Pre-trained Model Weights

Below is a table showcasing model names and their corresponding download links:

| Model Name | Download Link |
|------------|---------------|
| Model 1 (From the original pre-training) | [Link](https://github.com/dandelin/ViLT/releases/download/200k/vilt_200k_mlm_itm.ckpt) |
| Model 2 (With additional large-scale phrase grounding pre-training) | [Link](https://drive.google.com/file/d/1NScQp-QopP7iTU0j9bt7BY2lReu0sYhN/view?usp=sharing) |

### 2. Fine-tuning Models 

Before starting with the commands, familiarize yourself with the following key variables:

- **DATA_BINARY_PATH**: Path to the folder containing data binaries. To prepare the data binary, refer to this [README](https://github.com/lil-lab/phrase_grounding_working/tree/main/src/preprocessing). 
  * Example: `data/touchdown_sdr/binary`
  
- **PRETRAINED_MODEL_PATH**: Path to the pre-trained model checkpoint weights obtained from the previous step.
  * Example: `checkpoints/tangram-refgame-viltaligner-pretrained_seed0_from_epoch=19-step=392299/version_0/checkpoints/epoch=97-step=20580.ckpt`
  
- **EXP_NAME**: Name for your experiments.
  * Example: `touchdown_sdr_pretrained_grounding`

Commands for fine-tuning:

- **With Grounding Annotations**:
  ```bash
  OMP_NUM_THREADS=1 MASTER_PORT=$RANDOM python run.py with data_root=DATA_BINARY_PATH vilt_aligner_touchdown_sdr per_gpu_batchsize=1 per_gpu_batchsize_eval=1 num_gpus=1 load_path=PRETRAINED_MODEL_PATH exp_name=EXP_NAME test_only=False use_ema=True load_from_ema=False use_grounding=True check_val_every_n_epoch=1
  ```

- **Without Grounding Annotations**:
  ```bash
  OMP_NUM_THREADS=1 MASTER_PORT=$RANDOM python run.py with data_root=DATA_BINARY_PATH vilt_aligner_touchdown_sdr per_gpu_batchsize=1 per_gpu_batchsize_eval=1 num_gpus=1 load_path=PRETRAINED_MODEL_PATH exp_name=EXP_NAME test_only=False use_ema=True load_from_ema=False use_grounding=False check_val_every_n_epoch=1
  ```

### 3. Testing Models 

To test your models, use the following command:

```bash
OMP_NUM_THREADS=1 MASTER_PORT=$RANDOM python run.py with data_root=DATA_BINARY_PATH vilt_aligner_touchdown_sdr per_gpu_batchsize=1 per_gpu_batchsize_eval=1 num_gpus=1 load_path=MODEL_PATH exp_name=EXP_NAME use_ema=True load_from_ema=False use_grounding=True test_update_db=True test_only=True disable_wandb=True
```

### Notes on Checkpoints

| Model Name | Download Link |
|------------|---------------|
| Additionally pre-trained & without grounding annotations | [Link](https://drive.google.com/file/d/1H5KkbHytz7U3vSm_p8L5_U_OZFFJXAGm/view?usp=sharing) |
| Additionally pre-trained & with grounding annotations | [Link](https://drive.google.com/file/d/1spL6i1tlcDHAMVbqCqD9lhNszGy8MSx5/view?usp=sharing) |

## Using MDETR

### 1. Downloading Pre-trained Model Weights

Below is a table showcasing model names and their corresponding download links:

| Model Name | Download Link |
|------------|---------------|
| Model 1 (From scratch) | [Link](https://drive.google.com/file/d/1f9GbpNn4A1Svjjgk9rAKiEr4gOAKnfju/view?usp=sharing) |
| Model 2 (With additional large-scale phrase grounding pre-training) | need to download |

### 2. Fine-tuning Models 

Before starting with the commands, familiarize yourself with the following key variables:

- **DATA_BINARY_PATH**: Path to the folder containing data binaries. To prepare the data binary, refer to this [README](https://github.com/lil-lab/phrase_grounding_working/tree/main/src/preprocessing). 
  * Example: `data/touchdown_sdr/binary`
  
- **PRETRAINED_MODEL_PATH**: Path to the pre-trained model checkpoint weights obtained from the previous step.
  * Example: `checkpoints/mdetr_random_initalization_processed.ckpt` (From scratch)
  * Example: `` (leave it empty With additional large-scale phrase grounding pre-training)
  
- **EXP_NAME**: Name for your experiments.
  * Example: `touchdown_sdr_pretrained_grounding`

Commands for fine-tuning:

- **With Grounding Annotations**:
  ```bash
  OMP_NUM_THREADS=1 MASTER_PORT=$RANDOM python run.py with data_root=DATA_BINARY_PATH mdetr_touchdown_sdr per_gpu_batchsize=2 num_gpus=4 per_gpu_batchsize_eval=1 batch_size=16 load_path=PRETRAINED_MODEL_PATH exp_name=EXP_NAME debug=False test_only=False use_grounding=True use_ema=True load_from_ema=False disable_wandb=False check_val_every_n_epoch=1 seed=100 weight_decay=1e-4 clip_max_norm=0.1
  ```

- **Without Grounding Annotations**:
  ```bash
  OMP_NUM_THREADS=1 MASTER_PORT=$RANDOM python run.py with data_root=DATA_BINARY_PATH mdetr_touchdown_sdr per_gpu_batchsize=2 num_gpus=4 per_gpu_batchsize_eval=1 batch_size=16 load_path=PRETRAINED_MODEL_PATH exp_name=EXP_NAME debug=False test_only=False use_grounding=False use_ema=True load_from_ema=False disable_wandb=False check_val_every_n_epoch=1 seed=100 weight_decay=1e-4 clip_max_norm=0.1
  ```

### 3. Testing Models 

To test your models, use the following command:

```bash
OMP_NUM_THREADS=1 MASTER_PORT=$RANDOM python run.py with data_root=DATA_BINARY_PATH mdetr_touchdown_sdr per_gpu_batchsize_eval=1 batch_size=1 load_path=MODEL_PATH exp_name=EXP_NAME use_ema=True load_from_ema=False use_grounding=True test_update_db=True test_only=True disable_wandb=True
```

### Notes on Checkpoints

| Model Name | Download Link |
|------------|---------------|
| Additionally pre-trained & without grounding annotations | [Link](https://drive.google.com/file/d/13WCEJ3RKM19ZYys7f06e_tR6e-r_5i2v/view?usp=sharing) |
| Additionally pre-trained & with grounding annotations | [Link](https://drive.google.com/file/d/1eTQrw5h8X50l-x6FMx95x7pLeB5dddJa/view?usp=drive_link) |
