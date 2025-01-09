import torch


def tokenize_and_align_labels(
        tokens,
        punct_labels,
        cap_labels,
        tokenizer,
        cap_label2id=None,
        punct_label2id=None,
        return_labels=True
):
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].squeeze(0)
    word_ids = encoding.word_ids(batch_index=0)
    subword_tokens = tokenizer.convert_ids_to_tokens(input_ids)

    if not return_labels:
        return encoding, subword_tokens

    punct_label_ids = []
    cap_label_ids = []

    for idx, word_id in enumerate(word_ids):
        if word_id is None:
            punct_label_ids.append(-100)
            cap_label_ids.append(-100)
        else:
            if word_ids[idx] != word_ids[idx - 1] if idx > 0 else True:
                cap_label_ids.append(cap_label2id[cap_labels[word_id]])
            else:
                cap_label_ids.append(cap_label2id["NO_CAP"])

            if idx + 1 < len(word_ids) and word_ids[idx + 1] == word_id:
                punct_label_ids.append(punct_label2id["O"])
            else:

                punct_label_ids.append(punct_label2id[punct_labels[word_id]])

    return (
        encoding,
        torch.tensor(cap_label_ids, dtype=torch.long),
        torch.tensor(punct_label_ids, dtype=torch.long),
        subword_tokens,
    )
