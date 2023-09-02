from sacred import Experiment

ex = Experiment("Experiment")

def _loss_names(d):
    ret = {
        "touchdown_sdr": 0,
        "refgame": 0,
    }
    ret.update(d)
    return ret

@ex.config
def config():
    exp_name = ""
    seed = 0
    debug = False 
    disable_wandb = False
    model_cls = "vilt_aligner"
    fast_dev_run = False

    # task options
    use_grounding = True
    max_grounding_annotations = None
    touchdown_sdr_options = {
        "use_sdr_loss": True,
    }
    tangram_options = {
        "use_segmentation_mask": False,
    }
    refgame_options = {
        "num_distractors": 5,
        "topk": 20,
        "train_sampling_method": "sample_topk",
        "test_sampling_method": "top_k",
        "cls_loss_alpha": 3,
        "train_fix_contexts": False,
        "train_fix_images": False,
    }
    use_touchdown_loss_for_cls = False

    # training
    datasets = []
    loss_names = _loss_names({})
    resume_from = None
    wandb_run_id = None
    batch_size = 16
    eval_batch_size = 16
    per_gpu_batchsize = 1  
    per_gpu_batchsize_eval = 1
    val_check_interval = 1.0
    check_val_every_n_epoch = 1
    data_root = ""
    load_path = ""
    num_gpus = 1
    num_nodes = 1
    num_workers = 4
    precision = 16
    save_top_k = -1
    log_dir = "result/vilt/checkpoints"

    # testing / analysis
    test_only = False
    test_update_db = False

    # image settings
    train_transform_keys = ["mdetr_randaug"]
    val_transform_keys = ["pixelbert_noresize"]
    image_size = 384
    max_image_len = -1
    patch_size = 32 

    # text Setting
    max_text_len = 80 
    tokenizer = "bert-base-uncased"
    vocab_size = 30522
    additional_tokens = []

    # optimizer Setting
    optim_type = "adamw"
    learning_rate = 1e-4
    weight_decay = 0.01
    decay_power = 1
    clip_max_norm = 0.
    max_epoch = 100
    max_steps = None
    warmup_steps = 0.01
    end_lr = 0
    lr_mult = 1  
    use_ema = False
    load_from_ema = True

    # ViLT-Aligner settings
    vit = "vit_base_patch32_384"
    hidden_size = 768
    num_heads = 12
    num_layers = 12
    mlp_ratio = 4
    drop_rate = 0.1
    vilt_split_detection_head = False
    
    # MDETR settings
    contrastive_align_loss = True
    aux_loss = False
    dec_layers = 6
    cost_class = 1
    cost_bbox = 5
    cost_giou = 2
    eos_coef = 0.1
    temperature_NCE = 0.07
    ce_loss_coef = 1
    contrastive_align_loss_coef = 1
    giou_loss_coef = 2
    bbox_loss_coef = 5
    touchdown_sdr_coef = 9
    cls_coef = 25
    learning_rate_backbone = 1.4e-5
    learning_rate_text_encoder = 7e-5
    lr_drop = 40
    mdetr_weights_init = {
        "initialize_sdr_head_from_bbox_embed": True,
    }
    task_head_setup = {
        "use_shared_head": False,
    }

    # Ablation
    training_phrase_grounding_ratio = 1.0
    training_grounding_annotated_image_ratio = 1.0


@ex.named_config
def vilt_aligner_touchdown_sdr():
    model_cls = "vilt_aligner"
    loss_names = _loss_names({"touchdown_sdr": 1})
    datasets = ["touchdown_sdr"]
    image_size = 3712
    train_transform_keys = ["pixelbert_noresize"]
    val_transform_keys = ["pixelbert_noresize"]

@ex.named_config
def mdetr_touchdown_sdr():
    model_cls = "mdetr"
    loss_names = _loss_names({"touchdown_sdr": 1})
    datasets = ["touchdown_sdr"]
    image_size = 3712
    train_transform_keys = ["mdetr_randaug"]
    val_transform_keys = ["mdetr_touchdown_val"]
    precision = 16
    optim_type = "adamw"
    learning_rate = 1.4e-4
    weight_decay = 1e-4
    decay_power = "linear_with_warmup"
    max_epoch = 100
    lr_drop = 50

    # text settings
    max_text_len = 256 
    tokenizer = "roberta-base"

@ex.named_config
def vilt_aligner_f30k_refgame():
    model_cls = "vilt_aligner"
    loss_names = _loss_names({"refgame": 1})
    datasets = ["f30k_refgame"]
    image_size = 384
    train_transform_keys = ["pixelbert_square"]
    val_transform_keys = ["pixelbert_square"]
    max_epoch = 20
    val_check_interval = 0.5

@ex.named_config
def mdetr_f30k_refgame():
    model_cls = "mdetr"
    loss_names = _loss_names({"refgame": 1})
    datasets = ["f30k_refgame"]
    image_size = 384
    train_transform_keys = ["mdetr_square"]
    val_transform_keys = ["mdetr_square"]
    precision = 16
    optim_type = "adamw"
    learning_rate = 1.4e-4
    weight_decay = 1e-4
    decay_power = "linear_with_warmup"
    max_epoch = 20
    lr_drop = 10

    # text settings
    max_text_len = 256 
    tokenizer = "roberta-base"

@ex.named_config
def vilt_aligner_tangram_refgame():
    model_cls = "vilt_aligner"
    loss_names = _loss_names({"refgame": 1})
    datasets = ["tangram_refgame"]
    image_size = 224
    train_transform_keys = ["pixelbert_square"]
    val_transform_keys = ["pixelbert_square"]
    max_epoch = 20
    refgame_options = {
        "num_distractors": 9,
        "topk": 100,
        "train_sampling_method": "sample_topk",
        "test_sampling_method": "top_k",
        "cls_loss_alpha": 3
    }

@ex.named_config
def mdetr_tangram_refgame():
    model_cls = "mdetr"
    loss_names = _loss_names({"refgame": 1})
    datasets = ["tangram_refgame"]
    image_size = 224
    train_transform_keys = ["mdetr_square"]
    val_transform_keys = ["mdetr_square"]
    precision = 16
    optim_type = "adamw"
    learning_rate = 1.4e-4
    weight_decay = 1e-4
    decay_power = "linear_with_warmup"
    max_epoch = 20
    lr_drop = 10

    # text settings
    max_text_len = 256 
    tokenizer = "roberta-base"

    refgame_options = {
        "num_distractors": 9,
        "topk": 100,
        "train_sampling_method": "sample_topk",
        "test_sampling_method": "top_k",
    }
