import numpy as np
import unicodedata


def find_root(parse):
    # root node's head also == 0, so have to be removed
    for token in parse[1:]:
        if token.head == 0:
            return token.id
    return False


def _run_strip_accents(text):
    """Strips accents from a piece of text."""
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat == "Mn":
            continue
        output.append(char)
    return "".join(output)


def match_tokenized_to_untokenized(subwords, sentence):
    token_subwords = np.zeros(len(sentence))
    sentence = [_run_strip_accents(x) for x in sentence]
    token_ids, subwords_str, current_token, current_token_normalized = [-1] * len(subwords), "", 0, None
    for i, subword in enumerate(subwords):
        if subword in ["[CLS]", "[SEP]"]: continue

        while current_token_normalized is None:
            current_token_normalized = sentence[current_token].lower()

        if subword.startswith("[UNK]"):
            unk_length = int(subword[6:])
            subwords[i] = subword[:5]
            subwords_str += current_token_normalized[len(subwords_str):len(subwords_str) + unk_length]
        else:
            subwords_str += subword[2:] if subword.startswith("##") else subword
        if not current_token_normalized.startswith(subwords_str):
            return False

        token_ids[i] = current_token
        token_subwords[current_token] += 1
        if current_token_normalized == subwords_str:
            subwords_str = ""
            current_token += 1
            current_token_normalized = None

    assert current_token_normalized is None
    while current_token < len(sentence):
        assert not sentence[current_token]
        current_token += 1
    assert current_token == len(sentence)

    return token_ids
