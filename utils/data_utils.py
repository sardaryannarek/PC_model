from sklearn.model_selection import train_test_split


def split_train_test(input_text, test_size=0.2, random_state=42):

    sentences = input_text.strip().split("\n")

    train_data, test_data = train_test_split(
        sentences,
        test_size=test_size,
        random_state=random_state,
        shuffle=False  # Ensure no shuffling
    )

    return train_data, test_data
