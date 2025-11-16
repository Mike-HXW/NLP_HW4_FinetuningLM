import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Transformation: insert a short, neutral discourse phrase into the review.
    # Steps:
    # 1. Tokenize the original text.
    # 2. Sample a neutral phrase such as "to be honest" or "in my opinion".
    # 3. Insert the phrase either:
    #    - at the beginning of the review, OR
    #    - after a random comma, OR
    #    - at a random token boundary (if no comma is chosen/available).
    # 4. Detokenize back to a string and update example["text"].

    text = example.get("text", "")

    # If text is empty or very short, just return as is
    if not text:
        return example

    # Neutral, label-preserving discourse phrases
    neutral_phrases = [
        "to be honest",
        "in my opinion",
        "honestly",
        "I think",
        "for what it's worth",
        "by the way",
        "to be fair",
        "as a side note",
    ]

    # Tokenize the original review
    tokens = word_tokenize(text)

    # If tokenization fails or yields nothing, leave unchanged
    if len(tokens) == 0:
        return example

    # Choose a neutral phrase and tokenize it
    phrase = random.choice(neutral_phrases)
    phrase_tokens = word_tokenize(phrase)

    # Decide insertion position:
    # 50% chance to insert after a comma if any commas exist,
    # otherwise insert at a random token boundary (including start/end).
    comma_indices = [i for i, tok in enumerate(tokens) if tok == ","]
    if comma_indices and random.random() < 0.5:
        # Insert right after a random comma
        insert_pos = random.choice(comma_indices) + 1
    else:
        # Insert at a random boundary (0 .. len(tokens))
        insert_pos = random.randint(0, len(tokens))

    # Insert phrase tokens plus a comma after them to keep it natural: "Honestly, ..."
    new_tokens = tokens[:insert_pos] + phrase_tokens + [","] + tokens[insert_pos:]

    # Detokenize back into a single string
    detok = TreebankWordDetokenizer()
    new_text = detok.detokenize(new_tokens)

    example["text"] = new_text

    ##### YOUR CODE ENDS HERE ######

    return example
