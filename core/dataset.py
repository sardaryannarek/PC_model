from torch.utils.data import Dataset
from utils.text_processing import split_text_into_chunks, preprocess_text_and_labels
from utils.tokenizer_utils import tokenize_and_align_labels
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from core.config import MODEL_ID
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


class ModernBertPuncCapDataset(Dataset):

    def __init__(self, text, chunk_size, tokenizer, cap_label2id, punct_label2id, remove_extra_punctuation=False):

        self.samples = []
        chunks = list(split_text_into_chunks(text, chunk_size))

        for chunk in chunks:
            tokens, punct_labels, cap_labels = preprocess_text_and_labels(
                chunk, remove_extra_punctuation=remove_extra_punctuation
            )

            encoding, cap_label_ids, punct_label_ids, subword_tokens = tokenize_and_align_labels(
                tokens, punct_labels, cap_labels, tokenizer, cap_label2id, punct_label2id
            )

            self.samples.append({
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "punct_labels": punct_label_ids,
                "cap_labels": cap_label_ids,
                "subword_tokens": subword_tokens,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]




def collate_fn(batch):

    input_ids_list = [item["input_ids"] for item in batch]
    attention_mask_list = [item["attention_mask"] for item in batch]
    punct_label_list = [item["punct_labels"] for item in batch]
    cap_label_list = [item["cap_labels"] for item in batch]
    subword_tokens_list = [item["subword_tokens"] for item in batch]

    padded_input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)

    padded_attention_masks = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)

    padded_punct_labels = pad_sequence(
        punct_label_list, batch_first=True, padding_value=-100
    )
    padded_cap_labels = pad_sequence(
        cap_label_list, batch_first=True, padding_value=-100
    )

    batch_out = {
        "input_ids": padded_input_ids,
        "attention_mask": padded_attention_masks,
        "punct_labels": padded_punct_labels,
        "cap_labels": padded_cap_labels,
        "subword_tokens": subword_tokens_list
    }

    return batch_out
