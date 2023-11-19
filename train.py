from copy import deepcopy

import torch
from tqdm import tqdm
from bigram import BigramLanguageModel
import argparse

from config import block_size, batch_size, eval_iters, lr, max_iters, eval_interval, device


def main():
    parser = argparse.ArgumentParser(prog='train.py', description='Train model.')
    parser.add_argument('inputfile', help='text file to train model with', type=str)
    parser.add_argument('outputfile', help='file to save the trained model to', type=str)
    parser.add_argument('--tokenizer', help='tokenizer to use: \'char\' or \'tiktoken\' ', type=str, default='char')
    parser.add_argument('--stop_early_steps', help='stop early if no improvement in loss over this many steps', type=int, default=0)
    args = parser.parse_args()

    #torch.manual_seed(1337)

    with open(args.inputfile, 'r', encoding='utf-8') as f:
        text = f.read()

    vocab_size = None
    data = None

    with torch.no_grad():
        if args.tokenizer == 'char':
            from char_tokenizer import CharTokenizer
            tokenizer = CharTokenizer()
            vocab_size = tokenizer.get_vocab_size()
            data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        elif args.tokenizer == 'tiktoken':
            import tiktoken
            encoding_name = 'p50k_base'
            enc = tiktoken.get_encoding(encoding_name)
            vocab_size = enc.n_vocab
            data = torch.tensor(enc.encode(text), dtype=torch.long)

    # split data into training and validation sets
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i + block_size] for i in ix])
        y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    model = BigramLanguageModel(vocab_size)
    m = model.to(device)

    print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")
    print(f'Number of tokens : {vocab_size}')
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    print("training...")
    best = None
    best_loss = float('inf')
    last_loss = float('inf')
    no_improvement = 0
    for iter in tqdm(range(max_iters)):
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if args.stop_early_steps > 0:
                if losses['val'] >= best_loss:
                    no_improvement += 1
                    if no_improvement >= args.stop_early_steps:
                        print(f"no improvement in {args.stop_early_steps} steps, stopping early")
                        break
                else:
                    no_improvement = 0
            if losses['val'] < best_loss:
                best_loss = losses['val']
                best = deepcopy(model.state_dict())
            last_loss = losses['val']
        xb, yb = get_batch('train')

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    torch.save(best, args.outputfile)
    print(f"saved model to {args.outputfile}")


if __name__ == '__main__':
    main()
