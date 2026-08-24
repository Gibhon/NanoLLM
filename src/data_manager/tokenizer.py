class CharTokenizer:
    def __init__(self, data:list):
        vocab = sorted(set(data))
        self.vocab_size = len(vocab)

        self.str_to_int = {ch:i for i, ch in enumerate(vocab)}
        self.int_to_str = {i:ch for i, ch in enumerate(vocab)}

    def encode(self, data:list)->list:
        return [self.str_to_int.get(ch, -1) for ch in data]

    def decode(self, token_ids):
        return [self.int_to_str.get(i, "") for i in token_ids]
