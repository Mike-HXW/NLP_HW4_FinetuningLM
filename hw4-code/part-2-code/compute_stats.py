"""
Compute simple dataset statistics for Q4 using the T5 tokenizer.

Outputs:
- Number of examples
- Mean NL length (tokens)
- Mean SQL length (tokens)
- Vocabulary size for NL
- Vocabulary size for SQL
"""

import os
import numpy as np
from transformers import T5TokenizerFast
from load_data import load_lines


def count_unique_tokens(texts, tokenizer):
    """Return the number of distinct token IDs that appear in the texts."""
    token_ids_set = set()
    for t in texts:
        ids = tokenizer.encode(t, add_special_tokens=True)
        token_ids_set.update(ids)
    return len(token_ids_set)


def main():
    tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")

    data_folder = "data"
    splits = ["train", "dev"]
    stats = {}

    for split in splits:
        nl_file = os.path.join(data_folder, f"{split}_pre.nl")
        sql_file = os.path.join(data_folder, f"{split}_pre.sql")

        nl_texts = load_lines(nl_file)
        sql_texts = load_lines(sql_file)

        # sequence lengths
        nl_tok = tokenizer(nl_texts, return_length=True, padding=False, truncation=False)
        sql_tok = tokenizer(sql_texts, return_length=True, padding=False, truncation=False)

        nl_lengths = nl_tok["length"]
        sql_lengths = sql_tok["length"]

        # vocab sizes
        nl_vocab = count_unique_tokens(nl_texts, tokenizer)
        sql_vocab = count_unique_tokens(sql_texts, tokenizer)

        stats[split] = {
            "num_examples": len(nl_texts),
            "mean_nl_len": np.mean(nl_lengths),
            "mean_sql_len": np.mean(sql_lengths),
            "vocab_nl": nl_vocab,
            "vocab_sql": sql_vocab,
        }

    # ===== Pretty-print =====
    print("\n" + "=" * 70)
    print("DATA STATISTICS (T5 Tokenizer)")
    print("=" * 70)
    print(f"{'Statistic':<40} {'Train':<15} {'Dev':<15}")
    print("-" * 70)

    print(f"{'Number of examples':<40} {stats['train']['num_examples']:<15} {stats['dev']['num_examples']:<15}")
    print(f"{'Mean NL length':<40} {stats['train']['mean_nl_len']:<15.2f} {stats['dev']['mean_nl_len']:<15.2f}")
    print(f"{'Mean SQL length':<40} {stats['train']['mean_sql_len']:<15.2f} {stats['dev']['mean_sql_len']:<15.2f}")
    print(f"{'Vocabulary size (NL)':<40} {stats['train']['vocab_nl']:<15} {stats['dev']['vocab_nl']:<15}")
    print(f"{'Vocabulary size (SQL)':<40} {stats['train']['vocab_sql']:<15} {stats['dev']['vocab_sql']:<15}")

    print("=" * 70)

    # ===== Markdown version =====
    print("\nMarkdown table:")
    print("```")
    print("| Statistic | Train | Dev |")
    print("|-----------|-------|-----|")
    print(f"| Number of examples | {stats['train']['num_examples']} | {stats['dev']['num_examples']} |")
    print(f"| Mean NL length | {stats['train']['mean_nl_len']:.2f} | {stats['dev']['mean_nl_len']:.2f} |")
    print(f"| Mean SQL length | {stats['train']['mean_sql_len']:.2f} | {stats['dev']['mean_sql_len']:.2f} |")
    print(f"| Vocabulary size (NL) | {stats['train']['vocab_nl']} | {stats['dev']['vocab_nl']} |")
    print(f"| Vocabulary size (SQL) | {stats['train']['vocab_sql']} | {stats['dev']['vocab_sql']} |")
    print("```")


if __name__ == "__main__":
    main()
