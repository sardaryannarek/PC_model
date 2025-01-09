import argparse
import torch
import os
import sys
sys.path.append('.')
from utils import read_file
from core.dataset import ModernBertPuncCapDataset
from transformers import AutoTokenizer
from core.config import cap_label2id, punct_label2id, CHUNK_SIZE, MODEL_ID


def load_text_files(train_file, test_file):
    train_text = read_file(train_file).strip()
    test_text = read_file(test_file).strip()
    return train_text, test_text


def create_and_save_datasets(train_text, test_text, tokenizer, output_dir, remove_extra_punctuation=True):

    os.makedirs(output_dir, exist_ok=True)

    train_dataset = ModernBertPuncCapDataset(
        text=train_text,
        chunk_size=CHUNK_SIZE,
        tokenizer=tokenizer,
        cap_label2id=cap_label2id,
        punct_label2id=punct_label2id,
        remove_extra_punctuation=remove_extra_punctuation,
    )
    test_dataset = ModernBertPuncCapDataset(
        text=test_text,
        chunk_size=CHUNK_SIZE,
        tokenizer=tokenizer,
        cap_label2id=cap_label2id,
        punct_label2id=punct_label2id,
        remove_extra_punctuation=remove_extra_punctuation
    )

    train_output_path = os.path.join(output_dir, "train.pt")
    test_output_path = os.path.join(output_dir, "test.pt")

    torch.save(train_dataset, train_output_path)
    torch.save(test_dataset, test_output_path)

    print(f"Datasets saved successfully:\n"
          f" - Train Dataset: {train_output_path}\n"
          f" - Test Dataset: {test_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load train/test .txt files and save them as .pt datasets.")
    parser.add_argument("--train_file", required=True, help="Path to the train .txt file.")
    parser.add_argument("--test_file", required=True, help="Path to the test .txt file.")
    parser.add_argument("--output_dir", default="data/pt_datasets",
                        help="Directory to save .pt files (default: data/pt_datasets)")
    parser.add_argument("--remove_extra_punctuation", type=bool, default=True,
                        help="Remove irrelevant punctuation during preprocessing (default: True).")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    train_text, test_text = load_text_files(args.train_file, args.test_file)

    create_and_save_datasets(
        train_text=train_text,
        test_text=test_text,
        tokenizer=tokenizer,
        output_dir=args.output_dir,
        remove_extra_punctuation=args.remove_extra_punctuation
    )
