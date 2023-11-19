import torch
import argparse
from bigram import BigramLanguageModel

from config import device
def main():
    parser = argparse.ArgumentParser(prog='model.py', description='Load model and generate text.')
    parser.add_argument('filename', help='file to load model from', type=str)
    parser.add_argument('--tokenizer', help='tokenizer to use: \'char\' or \'tiktoken\' ', type=str, default='char')
    parser.add_argument('--max_length', help='maximum length of generated text', type=int, default=500)

    args = parser.parse_args()

    vocab_size = None
    tokenizer = None

    if args.tokenizer == 'char':
        from char_tokenizer import CharTokenizer
        tokenizer = CharTokenizer()
        vocab_size = tokenizer.get_vocab_size()

    elif args.tokenizer == 'tiktoken':
        import tiktoken
        encoding_name = 'p50k_base'
        tokenizer = tiktoken.get_encoding(encoding_name)
        vocab_size = tokenizer.n_vocab

    model = BigramLanguageModel(vocab_size)
    model.load_state_dict(torch.load(args.filename))
    model.eval()

    m = model.to(device)
    print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    data = tokenizer.decode(m.generate(context, max_new_tokens=1500)[0].tolist())
    print(data)

if __name__ == '__main__':
    main()