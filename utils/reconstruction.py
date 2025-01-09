def reconstruct_sentence_with_word_ids(subword_tokens, word_ids, punct_labels, cap_labels, punct_id2symbol, cap_id2label):

    reconstructed = []
    current_word = ""
    current_word_id = None
    last_punct_label = None

    for token, w_id, punct, cap in zip(subword_tokens, word_ids, punct_labels, cap_labels):
        if w_id is None:
            continue

        if w_id != current_word_id:
            if current_word:
                if last_punct_label is not None and last_punct_label != -100:
                    current_word += punct_id2symbol[int(last_punct_label)]
                reconstructed.append(current_word)

            current_word = token.lstrip("##")
            current_word_id = w_id
            last_punct_label = punct

            if cap_id2label[int(cap)] == "CAP":
                current_word = current_word.capitalize()
        else:

            current_word += token.lstrip("##")
            last_punct_label = punct

    if current_word:
        if last_punct_label is not None and last_punct_label != -100:
            current_word += punct_id2symbol[int(last_punct_label)]
        reconstructed.append(current_word)

    return " ".join(reconstructed)