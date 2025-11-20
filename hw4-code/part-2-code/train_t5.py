import os
import pickle
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
import wandb
import time  

from t5_utils import (
    initialize_model,
    initialize_optimizer_and_scheduler,
    save_model,
    load_model_from_checkpoint,
    setup_wandb,
)
from transformers import T5TokenizerFast
from load_data import load_t5_data
from utils import compute_metrics, save_queries_and_records, read_queries, compute_records

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PAD_IDX = 0


def get_args():
    '''
    Arguments for training. You may choose to change or extend these as you see fit.
    '''
    parser = argparse.ArgumentParser(description='T5 training loop')

    # Model hyperparameters
    parser.add_argument('--finetune', action='store_true', help="Whether to finetune T5 or not")
    parser.add_argument(
        '--model_name',
        type=str,
        default='google-t5/t5-small',
        help='HF model name'
    )

    # Training hyperparameters
    parser.add_argument('--optimizer_type', type=str, default="AdamW", choices=["AdamW"],
                        help="What optimizer to use")
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)

    parser.add_argument('--scheduler_type', type=str, default="cosine", choices=["none", "cosine", "linear"],
                        help="Whether to use a LR scheduler and what type to use if so")
    parser.add_argument('--num_warmup_epochs', type=int, default=1,
                        help="How many epochs to warm up the learning rate for if using a scheduler")
    parser.add_argument('--max_n_epochs', type=int, default=10,
                        help="How many epochs to train the model for")
    parser.add_argument('--patience_epochs', type=int, default=4,
                        help="If validation performance stops improving, how many epochs should we wait before stopping?")

    parser.add_argument('--use_wandb', action='store_true',
                        help="If set, we will use wandb to keep track of experiments")
    
    parser.add_argument('--experiment_name', type=str, default='experiment',
                        help="How should we name this experiment?")
    

    # Data hyperparameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)

    # Decoding / generation hyperparameters
    parser.add_argument('--num_beams', type=int, default=5,
                        help="Number of beams to use for beam search during generation")
    parser.add_argument('--repetition_penalty', type=float, default=1.0,
                        help="Repetition penalty for generation (>1.0 discourages repetition)")
    parser.add_argument('--no_repeat_ngram_size', type=int, default=0,
                        help="If > 0, no n-gram of this size can be repeated during generation")
    parser.add_argument('--length_penalty', type=float, default=0.8,
                        help="Length penalty for beam search ( >1.0 favors shorter, <1.0 favors longer sequences)")
    parser.add_argument('--max_length', type=int, default=768,
                        help="Maximum length of generated sequences")

    # Freezing options (all default to no freezing)
    parser.add_argument('--freeze_embeddings', action='store_true',
                        help='Freeze shared/encoder/decoder embeddings.')
    parser.add_argument('--freeze_encoder', action='store_true',
                        help='Freeze the entire encoder stack.')
    parser.add_argument('--freeze_decoder', action='store_true',
                        help='Freeze the entire decoder stack.')
    parser.add_argument('--freeze_n_encoder_layers', type=int, default=0,
                        help='Freeze the first N encoder layers.')
    parser.add_argument('--freeze_n_decoder_layers', type=int, default=0,
                        help='Freeze the first N decoder layers.')

    args = parser.parse_args()
    return args


def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    best_f1 = -1
    epochs_since_improvement = 0

    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir

    gt_sql_path = os.path.join('data/dev_pre.sql')
    gt_record_path = os.path.join('records/ground_truth_dev.pkl')
    model_sql_path = os.path.join(f'results/t5_{model_type}_{args.experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{args.experiment_name}_dev.pkl')

    for epoch in range(args.max_n_epochs):
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"Epoch {epoch}: Average train loss was {tr_loss}")

        eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(
            args, model, dev_loader,
            gt_sql_path, model_sql_path,
            gt_record_path, model_record_path
        )
        print(f"Epoch {epoch}: Dev loss: {eval_loss}, Record F1: {record_f1}, Record EM: {record_em}, SQL EM: {sql_em}")
        print(f"Epoch {epoch}: {error_rate * 100:.2f}% of the generated outputs led to SQL errors")
        print('\n')
        
        if args.use_wandb:
            result_dict = {
                'train/loss': tr_loss,
                'dev/loss': eval_loss,
                'dev/record_f1': record_f1,
                'dev/record_em': record_em,
                'dev/sql_em': sql_em,
                'dev/error_rate': error_rate,
            }
            wandb.log(result_dict, step=epoch)

        if record_f1 > best_f1:
            best_f1 = record_f1
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        save_model(checkpoint_dir, model, best=False)
        if epochs_since_improvement == 0:
            save_model(checkpoint_dir, model, best=True)

        if epochs_since_improvement >= args.patience_epochs:
            break


def train_epoch(args, model, train_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss()

    for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(train_loader):
        optimizer.zero_grad()
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)

        logits = model(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            decoder_input_ids=decoder_input,
        )['logits']

        non_pad = decoder_targets != PAD_IDX
        loss = criterion(logits[non_pad], decoder_targets[non_pad])
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    return total_loss / total_tokens


def eval_epoch(args, model, dev_loader, gt_sql_pth, model_sql_path, gt_record_path, model_record_path):
    """
    Evaluate model on the development set.

    Returns (in this order, matching train()):
        eval_loss, record_f1, record_em, sql_em, error_rate
    """
    model.eval()
    device = next(model.parameters()).device

    total_loss = 0.0
    num_batches = 0
    all_pred_sql = []
    tokenizer = T5TokenizerFast.from_pretrained(args.model_name)

    with torch.no_grad():
        for batch in dev_loader:
            # batch is a tuple from normal_collate_fn:
            # (encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs)
            encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs = batch

            encoder_ids = encoder_ids.to(device)
            encoder_mask = encoder_mask.to(device)
            decoder_targets = decoder_targets.to(device)

            # ---- Forward pass for loss ----
            outputs = model(
                input_ids=encoder_ids,
                attention_mask=encoder_mask,
                labels=decoder_targets,
            )
            loss = outputs.loss
            total_loss += loss.item()
            num_batches += 1

            # ---- Generation for metrics ----
            gen_ids = model.generate(
                input_ids=encoder_ids,
                attention_mask=encoder_mask,
                max_length=args.max_length,
                num_beams=args.num_beams,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                length_penalty=args.length_penalty,
                early_stopping=True,
            )

            # Decode token IDs to SQL strings
            pred_sql_batch = tokenizer.batch_decode(
                gen_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            all_pred_sql.extend(pred_sql_batch)

    # Average dev loss
    avg_loss = total_loss / max(1, num_batches)

    # Save predicted SQL + predicted records to disk.
    save_queries_and_records(
        all_pred_sql,
        model_sql_path,
        model_record_path,
    )

    # Compute metrics: assume utils.compute_metrics returns
    # (sql_em, record_em, record_f1, error_rate_messages)
    sql_em, record_em, record_f1, model_error_msgs = compute_metrics(
        gt_sql_pth,
        model_sql_path,
        gt_record_path,
        model_record_path,
    )

    if model_error_msgs:
        error_rate = sum(1 for msg in model_error_msgs if msg) / len(model_error_msgs)
    else:
        error_rate = 0.0

    # IMPORTANT: order must match train():
    # eval_loss, record_f1, record_em, sql_em, error_rate
    return avg_loss, record_f1, record_em, sql_em, error_rate


def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    """
    Run inference on the test set and save the model's generated SQL queries
    and their associated records.

    test_loader batches come from test_collate_fn and have:
        encoder_ids, encoder_mask, initial_decoder_inputs
    """
    model.eval()
    device = next(model.parameters()).device
    tokenizer = T5TokenizerFast.from_pretrained(args.model_name)

    generated_queries = []

    # ---- overall timer ----
    start_time = time.time()
    num_batches = len(test_loader)
    print(f"[test_inference] Starting inference on {num_batches} batches...")

    # Per-batch timeout in seconds: if a single batch takes longer than this, we abort.
    MAX_BATCH_TIME = 60.0

    with torch.no_grad():
        for batch_idx, (encoder_ids, encoder_mask, initial_decoder_inputs) in enumerate(
            tqdm(test_loader, total=num_batches, desc="Test inference", ncols=80)
        ):
            batch_start = time.time()

            # Move to device
            encoder_ids = encoder_ids.to(device)
            encoder_mask = encoder_mask.to(device)
            # initial_decoder_inputs is not needed directly if we rely on model.generate

            # Generation settings using CLI arguments
            gen_kwargs = {
                "max_length": args.max_length,
                "num_beams": args.num_beams,
                "repetition_penalty": args.repetition_penalty,
                "no_repeat_ngram_size": args.no_repeat_ngram_size,
                "length_penalty": args.length_penalty,
                "early_stopping": True,
            }

            generated = model.generate(
                input_ids=encoder_ids,
                attention_mask=encoder_mask,
                **gen_kwargs,
            )

            # Decode generated token IDs to SQL strings
            for gen in generated:
                sql = tokenizer.decode(gen, skip_special_tokens=True)
                generated_queries.append(sql)

            # Per-batch timing and timeout check
            batch_duration = time.time() - batch_start
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                elapsed = time.time() - start_time
                print(
                    f"[test_inference] Processed {batch_idx + 1}/{num_batches} batches "
                    f"in {elapsed:.1f}s (last batch {batch_duration:.1f}s)"
                )

            if batch_duration > MAX_BATCH_TIME:
                print(
                    f"[test_inference][ERROR] Batch {batch_idx} took {batch_duration:.1f}s "
                    f"(>{MAX_BATCH_TIME:.1f}s). Aborting test inference early."
                )
                break

    total_infer_time = time.time() - start_time
    print(
        f"[test_inference] Finished generation (or aborted) after {total_infer_time:.1f}s. "
        f"Now saving results..."
    )

    # ---- time the saving step separately ----
    save_start = time.time()
    print(
        f"[test_inference] Saving {len(generated_queries)} generated queries "
        f"to {model_sql_path} and {model_record_path}"
    )

    save_queries_and_records(generated_queries, model_sql_path, model_record_path)

    save_time = time.time() - save_start
    print(f"[test_inference] Finished saving in {save_time:.1f}s.")
    print(
        f"[test_inference] Total test_inference time: "
        f"{(time.time() - start_time):.1f}s (generation + saving)."
    )


def main():
    # Get key arguments
    args = get_args()
    print(f"[DEBUG] args.finetune = {args.finetune}")
    if args.use_wandb:
        # Recommended: Using wandb (or tensorboard) for result logging can make experimentation easier
        setup_wandb(args)

    # Load the data and the model
    train_loader, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size)
    model = initialize_model(args)
    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))

    # Train 
    train(args, model, train_loader, dev_loader, optimizer, scheduler)

    # Evaluate
    model = load_model_from_checkpoint(args, best=True)
    model.eval()
    
    # Dev set
    model_type = 'ft' if args.finetune else 'scr'
    gt_sql_path = os.path.join('data/dev_pre.sql')
    gt_record_path = os.path.join('records/ground_truth_dev.pkl')
    model_sql_path = os.path.join(f'results/t5_{model_type}_{args.experiment_name}_dev_pre.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{args.experiment_name}_dev.pkl')
    dev_loss, dev_record_em, dev_record_f1, dev_sql_em, dev_error_rate = eval_epoch(
        args, model, dev_loader,
        gt_sql_path, model_sql_path,
        gt_record_path, model_record_path
    )
    print(f"Dev set results: Loss: {dev_loss}, Record F1: {dev_record_f1}, Record EM: {dev_record_em}, SQL EM: {dev_sql_em}")
    print(f"Dev set results: {dev_error_rate * 100:.2f}% of the generated outputs led to SQL errors")

    # Test set
    model_sql_path = os.path.join(f'results/t5_{model_type}_{args.experiment_name}_test.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{args.experiment_name}_test.pkl')
    test_inference(args, model, test_loader, model_sql_path, model_record_path)


if __name__ == "__main__":
    main()
