import sys
sys.path.append(".")

import yaml
import torch
from collections import defaultdict
from transformers import AutoTokenizer

from core.PC_NET import PCNet
from utils.tokenizer_utils import tokenize_and_align_labels
from utils.reconstruction import reconstruct_sentence_with_word_ids
from utils.file_utils import read_file
from utils.text_processing import post_processing
from core.config import MODEL_ID, punct_id2symbol, cap_id2label



with open("scripts/always_predictable.yml", "r") as f:
    yml_file = yaml.safe_load(f)
always_capital = yml_file["always_capitalized_tokens"]
always_period = yml_file["always_period_abbreviations"]


def chunk_text_by_words(all_words, window_size=49, stride=25):
    chunks = []
    n = len(all_words)
    i = 0
    while i < n:
        start = i
        end = min(i + window_size, n)
        chunk_words = all_words[start:end]
        chunks.append({
            "start": start,
            "end": end,
            "words": chunk_words
        })
        i += stride
    return chunks

def process_chunk(model, tokenizer, chunk_words):

    encoding, subword_tokens = tokenize_and_align_labels(
        tokens=chunk_words,
        punct_labels=None,
        cap_labels=None,
        tokenizer=tokenizer,
        punct_label2id=None,
        cap_label2id=None,
        return_labels=False
    )

    # Move to device & run inference
    input_ids = encoding["input_ids"].to(model.device)
    attention_mask = encoding["attention_mask"].to(model.device)
    with torch.no_grad():
        punct_logits_batch, cap_logits_batch = model(input_ids, attention_mask)

    punct_logits_batch = punct_logits_batch.squeeze(0)
    cap_logits_batch = cap_logits_batch.squeeze(0)

    word_ids = encoding.word_ids(batch_index=0)

    word_level_preds = [None] * len(chunk_words)

    for sub_idx, w_id in enumerate(word_ids):
        if w_id is not None:
            word_level_preds[w_id] = (
                w_id,
                chunk_words[w_id],
                punct_logits_batch[sub_idx],
                cap_logits_batch[sub_idx]
            )

    return [p for p in word_level_preds if p is not None]


def process_text(model, tokenizer, text, window_size=49, stride=25):

    all_words = text.split()
    total_words = len(all_words)

    chunks = chunk_text_by_words(all_words, window_size, stride)

    # Dictionary to accumulate (punct_logits, cap_logits)
    # for each global word index
    accumulated_logits = defaultdict(list)

    # Process each chunk
    for ch in chunks:
        chunk_words = ch["words"]
        start_idx = ch["start"]

        word_preds = process_chunk(model, tokenizer, chunk_words)

        for (local_idx, w_str, p_logits, c_logits) in word_preds:
            global_idx = start_idx + local_idx
            accumulated_logits[global_idx].append((p_logits, c_logits))


    final_preds = {}

    for g_idx in range(total_words):
        if g_idx not in accumulated_logits:
            continue

        p_logits_list = [item[0] for item in accumulated_logits[g_idx]]
        c_logits_list = [item[1] for item in accumulated_logits[g_idx]]

        p_logits_avg = torch.mean(torch.stack(p_logits_list), dim=0)
        c_logits_avg = torch.mean(torch.stack(c_logits_list), dim=0)

        punct_label = torch.argmax(p_logits_avg).item()
        cap_label   = torch.argmax(c_logits_avg).item()

        final_preds[g_idx] = (all_words[g_idx], punct_label, cap_label)


    sorted_indices = sorted(final_preds.keys())


    subword_tokens = []
    word_ids = []
    punct_labels = []
    cap_labels = []

    for g_idx in sorted_indices:
        w_str, p_label, c_label = final_preds[g_idx]
        subword_tokens.append(w_str)
        word_ids.append(g_idx)
        punct_labels.append(p_label)
        cap_labels.append(c_label)


    cap_labels, punct_labels = post_processing(
        subword_tokens=subword_tokens,
        word_ids=word_ids,
        punct_preds=punct_labels,
        cap_preds=cap_labels,
        always_capital=always_capital,
        always_period=always_period
    )

    reconstructed_text = reconstruct_sentence_with_word_ids(
        subword_tokens=subword_tokens,
        word_ids=word_ids,
        punct_labels=punct_labels,
        cap_labels=cap_labels,
        punct_id2symbol=punct_id2symbol,
        cap_id2label=cap_id2label
    )

    return reconstructed_text


def main(checkpoint_dir, input_txt_file, window_size=49, stride=25, output_txt_file=None):

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = PCNet.load_from_checkpoint(checkpoint_dir)
    model.eval()
    print("Model loaded successfully.")

    text = read_file(input_txt_file)
    print("Text loaded successfully.")

    reconstructed_text = process_text(model, tokenizer, text, window_size, stride)
    print("Text processed successfully.")

    if output_txt_file is not None:
        with open(output_txt_file, "w", encoding="utf-8") as f:
            f.write(reconstructed_text)
        print(f"Output written to: {output_txt_file}")
    else:
        print("Reconstructed text:\n", reconstructed_text)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inference script for punctuation and capitalization model (with overlap-averaging).")
    parser.add_argument("--checkpoint_dir", required=True, help="Path to the model checkpoint.")
    parser.add_argument("--input_txt_file", required=True, help="Path to the input text file.")
    parser.add_argument("--window_size", type=int, default=49, help="Number of *words* per chunk.")
    parser.add_argument("--stride", type=int, default=25, help="Overlap in terms of *words*.")
    parser.add_argument("--output_txt_file", required=False, help="Path to the output text file.")
    args = parser.parse_args()

    main(
        checkpoint_dir=args.checkpoint_dir,
        input_txt_file=args.input_txt_file,
        window_size=args.window_size,
        stride=args.stride,
        output_txt_file=args.output_txt_file
    )
