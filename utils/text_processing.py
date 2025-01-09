import re
from core.config import PUNCT_SYMBOLS, punct_label2id


def is_decimal_number(word):
    decimal_pattern = re.compile(r"^\d+\.\d+$")
    return bool(decimal_pattern.match(word))


def split_text_into_chunks(text, chunk_size):
    words = text.strip().split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i + chunk_size])


def custom_split(text, punc_marks=tuple(PUNCT_SYMBOLS.keys()), remove_extra_punctuation=False):
    # 1) Find tokens: decimal numbers, words, or single punctuation
    raw_tokens = re.findall(r"[+-]?\d+(?:\.\d+)?|\w+|[^\w\s]", text)

    results = []
    any_punc_pattern = r'[^\w\s]+$'

    for token in raw_tokens:
        if is_decimal_number(token):
            results.append(token)

        elif remove_extra_punctuation:
            if token in punc_marks:
                results.append(token)
            else:
                base = re.sub(any_punc_pattern, '', token)
                if base:
                    results.append(base)

        else:
            results.append(token)

    return results



def preprocess_text_and_labels(chunk, remove_extra_punctuation=False):
    to_tokens = []
    punct_labels = []
    cap_labels = []

    words = custom_split(chunk, tuple(PUNCT_SYMBOLS.keys()), remove_extra_punctuation)
    for word in words:
        if word in {".", ",", "?"}:
            if to_tokens:
                punct_labels[-1] = word
            continue
        to_tokens.append(word.lower())

        cap_label = "CAP" if bool(re.search(r"[A-Z]", word)) else "NO_CAP"
        cap_labels.append(cap_label)

        punct_labels.append("O")

    return to_tokens, punct_labels, cap_labels


def post_processing(subword_tokens, word_ids, punct_preds, cap_preds, always_capital=None, always_period=None):


    for i in range(1, len(subword_tokens) - 1):
        separate_word = (word_ids[i] != word_ids[i - 1] and word_ids[i] != word_ids[i + 1])
        if separate_word:
            current_token = subword_tokens[i]

            if current_token in always_capital:
                cap_preds[i] = 1

            if current_token in always_period:
                punct_preds[i] = punct_label2id["."]

    return cap_preds, punct_preds


