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
    if not text:
        return example

    tokens = word_tokenize(text)
    if len(tokens) == 0:
        return example

    qwerty_neighbors = {
        "q": "w",
        "w": "qe",
        "e": "wr",
        "r": "et",
        "t": "ry",
        "y": "tu",
        "u": "yi",
        "i": "uo",
        "o": "ip",
        "p": "o",
        "a": "s",
        "s": "ad",
        "d": "sf",
        "f": "dg",
        "g": "fh",
        "h": "gj",
        "j": "hk",
        "k": "jl",
        "l": "k",
        "z": "x",
        "x": "zc",
        "c": "xv",
        "v": "cb",
        "b": "vn",
        "n": "bm",
        "m": "n",
    }

    detok = TreebankWordDetokenizer()
    new_tokens = []

    for tok in tokens:
        if tok.isalpha():
            lowered = tok.lower()

            if random.random() < 0.05:
                continue  

            if random.random() < 0.20:
                synsets = wordnet.synsets(lowered)
                lemmas = []
                for s in synsets:
                    for l in s.lemmas():
                        lemma_name = l.name().replace("_", " ")
                        if lemma_name.lower() != lowered and lemma_name.isalpha():
                            lemmas.append(lemma_name)
                if lemmas:
                    replacement = random.choice(lemmas)
                    if tok[0].isupper():
                        replacement = replacement.capitalize()
                    new_tokens.append(replacement)
                    continue  

            if random.random() < 0.30 and len(lowered) > 2:
                chars = list(lowered)
                idx = random.randint(1, len(chars) - 2)
                ch = chars[idx]
                if ch in qwerty_neighbors:
                    neighbors = qwerty_neighbors[ch]
                    chars[idx] = random.choice(neighbors)
                    corrupted = "".join(chars)
                    if tok[0].isupper():
                        corrupted = corrupted.capitalize()
                    new_tokens.append(corrupted)
                    continue

        new_tokens.append(tok)

    new_text = detok.detokenize(new_tokens)
    example["text"] = new_text

    ##### YOUR CODE ENDS HERE ######

    return example