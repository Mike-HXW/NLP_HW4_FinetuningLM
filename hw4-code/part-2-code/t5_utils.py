import os

import torch

import transformers
from transformers import T5ForConditionalGeneration, T5Config
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
import wandb

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def setup_wandb(args):
    use_wandb = getattr(args, "use_wandb", False)
    if not use_wandb:
        return None

    project = getattr(args, "wandb_project", "t5_text2sql")
    run_name = getattr(args, "run_name", None)

    wandb_run = wandb.init(
        project=project,
        name=run_name,
        config=vars(args),
    )
    return wandb_run


def _get_model_name(args):
    """
    Helper to decide which HF checkpoint/config name to use.
    Falls back to google-t5/t5-small if args.model_name is not set.
    """
    return getattr(args, "model_name", "google-t5/t5-small")


def initialize_model(args):
    """
    Initialize the T5 model.

    If args.finetune is True:
        - Load pretrained weights from the given model_name.
    If args.finetune is False:
        - Initialize from the same model's config (random weights).
    Optionally applies parameter freezing based on args.
    """
    model_name = _get_model_name(args)

    if args.finetune:
        print(f"[initialize_model] Loading pretrained model: {model_name}")
        model = T5ForConditionalGeneration.from_pretrained(model_name)
    else:
        print(f"[initialize_model] Initializing from config (scratch): {model_name}")
        config = T5Config.from_pretrained(model_name)
        model = T5ForConditionalGeneration(config)

    # Optionally freeze subsets of the model before training
    freeze_model_parameters(model, args)

    model.to(DEVICE)
    _print_param_summary(model, prefix="[initialize_model]")
    return model


def freeze_model_parameters(model, args):
    """
    Apply simple freezing schemes based on args. All flags are optional:
      - freeze_embeddings: stop updating shared/encoder/decoder embeddings
      - freeze_encoder: freeze the entire encoder stack
      - freeze_decoder: freeze the entire decoder stack
      - freeze_n_encoder_layers: freeze first N encoder layers
      - freeze_n_decoder_layers: freeze first N decoder layers

    If these attributes are not present on args, sensible defaults are used.
    """
    freeze_embeddings = getattr(args, "freeze_embeddings", False)
    freeze_encoder = getattr(args, "freeze_encoder", False)
    freeze_decoder = getattr(args, "freeze_decoder", False)
    freeze_n_encoder_layers = getattr(args, "freeze_n_encoder_layers", 0)
    freeze_n_decoder_layers = getattr(args, "freeze_n_decoder_layers", 0)

    total_params = sum(p.numel() for p in model.parameters())

    # 1) Embeddings
    if freeze_embeddings:
        if hasattr(model, "shared") and model.shared is not None:
            model.shared.weight.requires_grad = False
        if hasattr(model, "encoder") and hasattr(model.encoder, "embed_tokens"):
            model.encoder.embed_tokens.weight.requires_grad = False
        if hasattr(model, "decoder") and hasattr(model.decoder, "embed_tokens"):
            model.decoder.embed_tokens.weight.requires_grad = False

    # 2) Whole encoder / decoder
    if freeze_encoder and hasattr(model, "encoder"):
        for p in model.encoder.parameters():
            p.requires_grad = False

    if freeze_decoder and hasattr(model, "decoder"):
        for p in model.decoder.parameters():
            p.requires_grad = False

    # 3) First N encoder layers
    if freeze_n_encoder_layers > 0 and hasattr(model, "encoder") and hasattr(model.encoder, "block"):
        n_layers = len(model.encoder.block)
        n_freeze = min(freeze_n_encoder_layers, n_layers)
        for i in range(n_freeze):
            for p in model.encoder.block[i].parameters():
                p.requires_grad = False
        if n_freeze > 0:
            print(f"[freeze_model_parameters] Frozen first {n_freeze}/{n_layers} encoder layers")

    # 4) First N decoder layers
    if freeze_n_decoder_layers > 0 and hasattr(model, "decoder") and hasattr(model.decoder, "block"):
        n_layers = len(model.decoder.block)
        n_freeze = min(freeze_n_decoder_layers, n_layers)
        for i in range(n_freeze):
            for p in model.decoder.block[i].parameters():
                p.requires_grad = False
        if n_freeze > 0:
            print(f"[freeze_model_parameters] Frozen first {n_freeze}/{n_layers} decoder layers")

    # Small summary
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    if frozen_params > 0:
        print("[freeze_model_parameters] Parameter stats:")
        print(f"  Total:     {total_params:,}")
        print(f"  Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        print(f"  Frozen:    {frozen_params:,} ({100*frozen_params/total_params:.2f}%)")


def _print_param_summary(model, prefix=""):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{prefix} Model parameters: total={total_params:,}, "
          f"trainable={trainable_params:,} "
          f"({100*trainable_params/total_params:.2f}%)")


def mkdir(dirpath):
    if not os.path.exists(dirpath):
        try:
            os.makedirs(dirpath)
        except FileExistsError:
            pass


def save_model(checkpoint_dir, model, best):
    """
    Save model checkpoint (just the state_dict) so we can resume / evaluate later.

    best == True  -> best.pt
    best == False -> last.pt
    """
    mkdir(checkpoint_dir)

    filename = "best.pt" if best else "last.pt"
    ckpt_path = os.path.join(checkpoint_dir, filename)

    state = {
        "model_state_dict": model.state_dict(),
    }
    torch.save(state, ckpt_path)
    print(f"[save_model] Saved checkpoint to {ckpt_path}")


def load_model_from_checkpoint(args, best):
    """
    Recreate the model (from pretrained or config) and load a saved checkpoint.

    If the checkpoint is missing or clearly corrupted, we continue with
    the freshly initialized model instead of crashing.
    """
    model = initialize_model(args)

    checkpoint_dir = getattr(args, "checkpoint_dir", "checkpoints")
    filename = "best.pt" if best else "last.pt"
    ckpt_path = os.path.join(checkpoint_dir, filename)

    if not os.path.exists(ckpt_path):
        print(f"[load_model_from_checkpoint] No checkpoint found at {ckpt_path}, "
              f"using freshly initialized model.")
        model.to(DEVICE)
        return model

    file_size = os.path.getsize(ckpt_path)
    if file_size < 1024:  # 1 KB threshold to catch obviously bad files
        print(f"[load_model_from_checkpoint] Warning: checkpoint {ckpt_path} is very small "
              f"({file_size} bytes). Skipping load and using fresh model.")
        model.to(DEVICE)
        return model

    try:
        state = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(state["model_state_dict"])
        print(f"[load_model_from_checkpoint] Loaded checkpoint from {ckpt_path} "
              f"({file_size / (1024*1024):.2f} MB)")
    except Exception as e:
        print(f"[load_model_from_checkpoint] ERROR loading checkpoint from {ckpt_path}: {e}")
        print("  Proceeding with a freshly initialized model instead.")

    model.to(DEVICE)
    _print_param_summary(model, prefix="[load_model_from_checkpoint]")
    return model


def initialize_optimizer_and_scheduler(args, model, epoch_length):
    optimizer = initialize_optimizer(args, model)
    scheduler = initialize_scheduler(args, optimizer, epoch_length)
    return optimizer, scheduler


def initialize_optimizer(args, model):
    # Same idea as before: apply weight decay only to non-LayerNorm, non-bias params
    decay_parameters = get_parameter_names(model, transformers.pytorch_utils.ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    if args.optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            eps=1e-8,
            betas=(0.9, 0.999),
        )
    else:
        raise NotImplementedError(f"Unsupported optimizer_type: {args.optimizer_type}")

    return optimizer


def initialize_scheduler(args, optimizer, epoch_length):
    num_training_steps = epoch_length * args.max_n_epochs
    num_warmup_steps = epoch_length * args.num_warmup_epochs

    if args.scheduler_type == "none":
        return None
    elif args.scheduler_type == "cosine":
        return transformers.get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )
    elif args.scheduler_type == "linear":
        return transformers.get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )
    else:
        raise NotImplementedError(f"Unsupported scheduler_type: {args.scheduler_type}")


def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result
