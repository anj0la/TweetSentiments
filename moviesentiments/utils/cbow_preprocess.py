def generate_context_target_pairs(text: list, vocab: dict, window_size: int = 2) -> list:
    """
    Generates context-target pairs for CBOW without padding, ensuring each target has a full context.

    Args:
        text (list): List of tokenized sentences, where each sentence is a list of words.
        vocab (dict): The vocabulary with token-to-index mappings.
        window_size (int, optional): The context window size. Defaults to 2.

    Returns:
        list: Generated context-target pairs with indices for model input.
    """
    context_target_pairs = []
    unk_index = len(vocab)

    for sentence in text:
        sequence = [vocab.get(word, unk_index) for word in sentence]

        # Only generate pairs when there are enough context words on both sides
        for i in range(window_size, len(sequence) - window_size):
            # Get 2 words to the left and 2 to the right
            left_context = sequence[i - window_size:i]
            right_context = sequence[i + 1:i + window_size + 1]
            context_words = left_context + right_context

            target = sequence[i]
            context_target_pairs.append((context_words, target))

    return context_target_pairs





