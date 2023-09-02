# Flickr30k Entites Experiments Guide

Welcome to the Kilogram Experiments Guide. Follow the steps below to fine-tune and test models using the ViLT-Aligner.

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
  * Example: `data/f30k/binary`
  
- **PRETRAINED_MODEL_PATH**: Path to the pre-trained model checkpoint weights obtained from the previous step.
  * Example: `checkpoints/tangram-refgame-viltaligner-pretrained_seed0_from_epoch=19-step=392299/version_0/checkpoints/epoch=97-step=20580.ckpt`
  
- **EXP_NAME**: Name for your experiments.
  * Example: `f30k_pretrained_grounding`

Commands for fine-tuning:

- **With Grounding Annotations**:
  ```bash
  OMP_NUM_THREADS=1 MASTER_PORT=$RANDOM python run.py with data_root=DATA_BINARY_PATH vilt_aligner_f30k_refgame per_gpu_batchsize=2 per_gpu_batchsize_eval=1 batch_size=32 num_gpus=1 load_path=PRETRAINED_MODEL_PATH exp_name=EXP_NAME test_only=False use_ema=True load_from_ema=False use_grounding=True use_touchdown_loss_for_cls=True max_epoch=100 check_val_every_n_epoch=1
  ```

- **Without Grounding Annotations**:
  ```bash
  OMP_NUM_THREADS=1 MASTER_PORT=$RANDOM python run.py with data_root=DATA_BINARY_PATH vilt_aligner_f30k_refgame per_gpu_batchsize=2 per_gpu_batchsize_eval=1 batch_size=32 num_gpus=1 load_path=PRETRAINED_MODEL_PATH exp_name=EXP_NAME test_only=False use_ema=True load_from_ema=False use_grounding=False use_touchdown_loss_for_cls=True max_epoch=100 check_val_every_n_epoch=1
  ```

### 3. Testing Models 

To test your models, use the following command:

```bash
OMP_NUM_THREADS=1 MASTER_PORT=$RANDOM python run.py with data_root=DATA_BINARY_PATH  vilt_aligner_f30k_refgame per_gpu_batchsize=4 per_gpu_batchsize_eval=1 batch_size=32 num_gpus=1 load_path=MODEL_PATH exp_name=EXP_NAME use_ema=True load_from_ema=False use_grounding=True use_touchdown_loss_for_cls=True test_update_db=True test_only=True disable_wandb=True
```

### Notes on Checkpoints

| Model Name | Download Link |
|------------|---------------|
| Additionally pre-trained & without grounding annotations | [Link](https://drive.google.com/file/d/1sQGwY-U0IUakc32JA4KAfbkm-kINodUc/view?usp=sharing) |
| Additionally pre-trained & with grounding annotations | [Link](https://drive.google.com/file/d/1FaIfMpixlQnTSADwVS8IuP-oBRwKSnRq/view?usp=sharing) |

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
  * Example: `data/f30k/binary`
  
- **PRETRAINED_MODEL_PATH**: Path to the pre-trained model checkpoint weights obtained from the previous step.
  * Example: `checkpoints/mdetr_random_initalization_processed.ckpt` (From scratch)
  * Example: `` (leave it empty With additional large-scale phrase grounding pre-training)
  
- **EXP_NAME**: Name for your experiments.
  * Example: `f30k_pretrained_grounding`

Commands for fine-tuning:

- **With Grounding Annotations**:
  ```bash
  OMP_NUM_THREADS=1 MASTER_PORT=$RANDOM python run.py with data_root=DATA_BINARY_PATH mdetr_f30k_refgame per_gpu_batchsize=4 per_gpu_batchsize_eval=1 batch_size=32 num_gpus=1 load_path=PRETRAINED_MODEL_PATH exp_name=EXP_NAME debug=False test_only=False use_ema=True test_only=False use_grounding=True use_touchdown_loss_for_cls=True refgame_options.num_distractors=9 refgame_options.train_fix_contexts=False refgame_options.train_fix_images=False num_workers=4 learning_rate_backbone=4.2e-6 learning_rate_text_encoder=2.1e-5 learning_rate=4.2e-6 max_epoch=200 check_val_every_n_epoch=5 mdetr_weights_init.initialize_sdr_head_from_bbox_embed=True aux_loss=False
  ```

- **Without Grounding Annotations**:
  ```bash
  OMP_NUM_THREADS=1 MASTER_PORT=$RANDOM python run.py with data_root=DATA_BINARY_PATH mdetr_f30k_refgame per_gpu_batchsize=4 per_gpu_batchsize_eval=1 batch_size=32 num_gpus=1 load_path=PRETRAINED_MODEL_PATH exp_name=EXP_NAME debug=False test_only=False use_ema=True test_only=False use_grounding=False use_touchdown_loss_for_cls=True refgame_options.num_distractors=9 refgame_options.train_fix_contexts=False refgame_options.train_fix_images=False num_workers=4 learning_rate_backbone=4.2e-6 learning_rate_text_encoder=2.1e-5 learning_rate=4.2e-6 max_epoch=200 check_val_every_n_epoch=5 mdetr_weights_init.initialize_sdr_head_from_bbox_embed=True aux_loss=False
  ```

### 3. Testing Models 

To test your models, use the following command:

```bash
OMP_NUM_THREADS=1 MASTER_PORT=$RANDOM python run.py with data_root=DATA_BINARY_PATH mdetr_f30k_refgame per_gpu_batchsize=4 per_gpu_batchsize_eval=1 batch_size=32 num_gpus=1 load_path=MODEL_PATH exp_name=EXP_NAME use_ema=True use_grounding=True use_touchdown_loss_for_cls=True num_workers=4 check_val_every_n_epoch=5 mdetr_weights_init.initialize_sdr_head_from_bbox_embed=True test_update_db=True test_only=True disable_wandb=True
```

### Notes on Checkpoints

| Model Name | Download Link |
|------------|---------------|
| Additionally pre-trained & without grounding annotations | [Link](https://drive.google.com/file/d/19C68Gja0OY1hCSabf1rDdfCxXhAAVp5h/view?usp=sharing) |
| Additionally pre-trained & with grounding annotations | [Link](https://drive.google.com/file/d/1ioByGEo4UESTO5VOg4dCS9Oe2jaIMVYb/view?usp=sharing) |
