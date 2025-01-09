from core.dataset import ModernBertPuncCapDataset
from utils import read_file
from core.config import CHUNK_SIZE, MODEL_ID, cap_label2id, punct_label2id
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset
from utils.text_processing import split_text_into_chunks, preprocess_text_and_labels
from utils.tokenizer_utils import tokenize_and_align_labels
import numpy as np


def load_text_files(train_file):
    train_text = read_file(train_file).strip()
    return train_text

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
remove_extra_punctuation = False
train_file = '../data/processed/train_train_val_pc.txt'
train_text = load_text_files(train_file)

def tokenize_custom(tokens, punct_labels, cap_labels, tokenizer, cap_label2id, punct_label2id, return_labels=True):
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].squeeze(0)
    word_ids = encoding.word_ids(batch_index=0)
    subword_tokens = tokenizer.convert_ids_to_tokens(input_ids)


    return encoding, subword_tokens




def tokenize_len(text):
    samples = []
    chunks = list(split_text_into_chunks(text, CHUNK_SIZE))

    for chunk in chunks:
        tokens, punct_labels, cap_labels = preprocess_text_and_labels(
            chunk, remove_extra_punctuation=remove_extra_punctuation
        )

        encoding, subword_tokens = tokenize_custom(
            tokens, punct_labels, cap_labels, tokenizer, cap_label2id, punct_label2id
        )
        samples.append(len(subword_tokens))
    return np.array(samples)

ans = tokenize_len(train_text)
print(f"Max is {np.max(ans)}")
print(f"Min is {np.min(ans)}")
print(f"Mean is {np.mean(ans)}")
print(f"Std dev is {np.std(ans)}")
