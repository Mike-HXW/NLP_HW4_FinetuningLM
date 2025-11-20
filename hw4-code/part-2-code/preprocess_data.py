#!/usr/bin/env python3
import os
import re

DATA_DIR = "data"

# --------- Basic helpers --------- #

def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]

def write_lines(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

def normalize_whitespace(text: str) -> str:
    """
    Collapse all runs of whitespace (spaces, tabs, etc.) into a single space,
    and strip leading/trailing spaces.
    """
    return " ".join(text.split())


# --------- NL preprocessing --------- #

def preprocess_nl_line(line: str) -> str:
    # Lowercase
    line = line.lower()
    # Normalize whitespace
    line = normalize_whitespace(line)
    return line


# --------- SQL preprocessing --------- #

def preprocess_sql_line(line: str) -> str:
    # Strip leading/trailing whitespace first
    line = line.strip()

    # Remove trailing semicolons (one or more) at end of line
    # e.g., "SELECT * FROM t;;  " -> "SELECT * FROM t"
    line = re.sub(r";+\s*$", "", line)

    # Normalize whitespace
    line = normalize_whitespace(line)

    # Normalize spaces around '=':
    #   "a=1" -> "a = 1"
    #   "a = 1" -> "a = 1" (unchanged)
    # This will not affect '!=' or '>=' unless spaces are already weird.
    line = re.sub(r"\s*=\s*", " = ", line)

    # Final whitespace normalize again (just in case)
    line = normalize_whitespace(line)
    return line


# --------- Main pipeline --------- #

def preprocess_nl_file(input_name: str, output_name: str):
    in_path = os.path.join(DATA_DIR, input_name)
    out_path = os.path.join(DATA_DIR, output_name)

    print(f"[NL] Reading {in_path}")
    lines = read_lines(in_path)

    print(f"[NL] Preprocessing {len(lines)} lines...")
    processed = [preprocess_nl_line(l) for l in lines]

    print(f"[NL] Writing preprocessed NL to {out_path}")
    write_lines(out_path, processed)


def preprocess_sql_file(input_name: str, output_name: str):
    in_path = os.path.join(DATA_DIR, input_name)
    out_path = os.path.join(DATA_DIR, output_name)

    print(f"[SQL] Reading {in_path}")
    lines = read_lines(in_path)

    print(f"[SQL] Preprocessing {len(lines)} lines...")
    processed = [preprocess_sql_line(l) for l in lines]

    print(f"[SQL] Writing preprocessed SQL to {out_path}")
    write_lines(out_path, processed)


def main():
    # NL files: train/dev
    preprocess_nl_file("train.nl", "train_pre.nl")
    preprocess_nl_file("dev.nl", "dev_pre.nl")

    # SQL files: train/dev
    preprocess_sql_file("train.sql", "train_pre.sql")
    preprocess_sql_file("dev.sql", "dev_pre.sql")

    print("\n✓ Preprocessing complete.")


if __name__ == "__main__":
    main()
