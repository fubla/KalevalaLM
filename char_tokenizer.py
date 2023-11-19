
class CharTokenizer:
    def __init__(self):
        chars = ' !"\'*,-.:;?AEHIJKLMNOPRSTUVYadeghijklmnoprstuvyÄäö\n'
        print(''.join(chars))
        self.chars = [char for char in chars]
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text):
        encode = lambda s: [self.stoi[c] for c in s]
        return encode(text)

    def decode(self, tokens):
        decode = lambda l: ''.join([self.itos[i] for i in l])
        return decode(tokens)

    def get_vocab_size(self):
        return self.vocab_size

    def print_vocab(self):
        print(''.join(self.chars))