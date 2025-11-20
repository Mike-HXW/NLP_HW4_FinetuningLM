import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0


class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Class for performing data processing for the T5 model.

        Notes:
            * Uses the 'google-t5/t5-small' tokenizer to tokenize both
              the encoder and decoder output.
            * Uses pad_token_id as the decoder start token, matching
              T5ForConditionalGeneration's default decoder_start_token_id.
            * Behavior is different on the test set (no SQL labels).
        '''
        assert split in ["train", "dev", "test"]
        self.data_folder = data_folder
        self.split = split

        # T5 tokenizer
        self.tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")

        # Use pad_token_id as BOS / decoder start (matches T5 defaults)
        self.bos_token_id = self.tokenizer.pad_token_id

        # Storage for processed examples
        self.encoder_inputs = []           # list[Tensor]
        self.decoder_inputs = []           # list[Tensor], only for train/dev
        self.decoder_targets = []          # list[Tensor], only for train/dev
        self.initial_decoder_inputs = []   # list[Tensor] (shape [1]) for all splits
        self.texts = []                    # original NL questions (for debugging / logging)

        self.process_data(self.data_folder, self.split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        """
        Reads NL questions and SQL queries from disk and tokenizes them.

        For train/dev:
            data/train_pre.nl   : preprocessed natural language questions
            data/dev_pre.nl
            data/train_pre.sql  : preprocessed SQL queries
            data/dev_pre.sql

        For test:
            data/test.nl        : natural language questions (no SQL labels)
        """
        nl_lines = None
        sql_lines = None

        # ----- Load NL (and SQL if not test) -----
        if split in ["train", "dev"]:
            # Preprocessed NL
            nl_path = os.path.join(data_folder, f"{split}_pre.nl")
            nl_lines = load_lines(nl_path)
            self.texts = nl_lines

            # Preprocessed SQL (NOW: {split}_pre.sql)
            sql_path = os.path.join(data_folder, f"{split}_pre.sql")
            sql_lines = load_lines(sql_path)
            assert len(nl_lines) == len(sql_lines), "Mismatch between NL and SQL lines."
        elif split == "test":
            # For test, we only have NL and we didn't create test_pre.nl
            nl_path = os.path.join(data_folder, "test.nl")
            nl_lines = load_lines(nl_path)
            self.texts = nl_lines
        else:
            raise ValueError(f"Unknown split: {split}")

        # ----- Tokenize and build tensors -----
        for i, nl in enumerate(nl_lines):
            # Encoder input: tokenize NL question with fixed max length
            enc_ids = tokenizer.encode(
                nl,
                add_special_tokens=True,  # T5 will add EOS
                max_length=512,
                truncation=True,
            )
            enc_tensor = torch.tensor(enc_ids, dtype=torch.long)
            self.encoder_inputs.append(enc_tensor)

            # Initial decoder input: always the BOS token (we use bos_token_id)
            init_dec = torch.tensor([self.bos_token_id], dtype=torch.long)
            self.initial_decoder_inputs.append(init_dec)

            if split != "test":
                sql = sql_lines[i]
                # Tokenize target SQL (includes EOS), with max length
                tgt_ids = tokenizer.encode(
                    sql,
                    add_special_tokens=True,
                    max_length=512,
                    truncation=True,
                )

                # Handle edge case: empty SQL
                if len(tgt_ids) == 0:
                    tgt_ids = [tokenizer.eos_token_id]

                # Teacher forcing shift:
                # decoder_input_ids = [BOS, y0, y1, ..., y_{n-2}]
                # decoder_target_ids = [y0, y1, ..., y_{n-1}]
                dec_in_ids = [self.bos_token_id] + tgt_ids[:-1]

                dec_in_tensor = torch.tensor(dec_in_ids, dtype=torch.long)
                tgt_tensor = torch.tensor(tgt_ids, dtype=torch.long)

                self.decoder_inputs.append(dec_in_tensor)
                self.decoder_targets.append(tgt_tensor)


    def __len__(self):
        return len(self.encoder_inputs)

    def __getitem__(self, idx):
        """
        Returns a dict so collate_fns can handle both train/dev and test cleanly.
        """
        item = {
            "encoder_ids": self.encoder_inputs[idx],
            "initial_decoder_input": self.initial_decoder_inputs[idx],
            "text": self.texts[idx],
        }

        if self.split != "test":
            item["decoder_inputs"] = self.decoder_inputs[idx]
            item["decoder_targets"] = self.decoder_targets[idx]
        else:
            item["decoder_inputs"] = None
            item["decoder_targets"] = None

        return item


def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Returns (in this order, matching train_t5.py):
        * encoder_ids: B x T
        * encoder_mask: B x T
        * decoder_inputs: B x T'
        * decoder_targets: B x T'
        * initial_decoder_inputs: B x 1
    '''
    encoder_seq_list = [b["encoder_ids"] for b in batch]
    decoder_in_seq_list = [b["decoder_inputs"] for b in batch]
    decoder_tgt_seq_list = [b["decoder_targets"] for b in batch]
    initial_dec_list = [b["initial_decoder_input"] for b in batch]

    # Pad encoder sequences
    encoder_ids = pad_sequence(
        encoder_seq_list,
        batch_first=True,
        padding_value=PAD_IDX,
    )  # (B, T)
    encoder_mask = (encoder_ids != PAD_IDX).long()  # (B, T)

    # Pad decoder sequences
    decoder_inputs = pad_sequence(
        decoder_in_seq_list,
        batch_first=True,
        padding_value=PAD_IDX,
    )  # (B, T')
    decoder_targets = pad_sequence(
        decoder_tgt_seq_list,
        batch_first=True,
        padding_value=PAD_IDX,
    )  # (B, T')

    # Stack initial decoder inputs (each is shape [1]) → (B, 1)
    initial_decoder_inputs = torch.stack(initial_dec_list, dim=0)

    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs


def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Returns:
        * encoder_ids: B x T
        * encoder_mask: B x T
        * initial_decoder_inputs: B x 1
    '''
    encoder_seq_list = [b["encoder_ids"] for b in batch]
    initial_dec_list = [b["initial_decoder_input"] for b in batch]

    encoder_ids = pad_sequence(
        encoder_seq_list,
        batch_first=True,
        padding_value=PAD_IDX,
    )
    encoder_mask = (encoder_ids != PAD_IDX).long()

    initial_decoder_inputs = torch.stack(initial_dec_list, dim=0)  # (B, 1)

    return encoder_ids, encoder_mask, initial_decoder_inputs


def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )
    return dataloader


def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines


def load_prompting_data(data_folder):
    """
    Load data for prompting experiments.

    Expects:
        data/train.nl : NL train questions
        data/train.sql : SQL train queries
        data/dev.nl   : NL dev questions
        data/dev.sql  : SQL dev queries
        data/test.nl  : NL test questions (no SQL labels)

    Returns:
        train_x, train_y, dev_x, dev_y, test_x
    """
    train_x = load_lines(os.path.join(data_folder, "train.nl"))
    train_y = load_lines(os.path.join(data_folder, "train.sql"))

    dev_x = load_lines(os.path.join(data_folder, "dev.nl"))
    dev_y = load_lines(os.path.join(data_folder, "dev.sql"))

    test_x = load_lines(os.path.join(data_folder, "test.nl"))

    return train_x, train_y, dev_x, dev_y, test_x
