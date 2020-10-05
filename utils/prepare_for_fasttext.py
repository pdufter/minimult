import argparse
from transformers import BertTokenizer
from tqdm import tqdm


def main_fake(args):
    tok = BertTokenizer.from_pretrained(args.vocab)
    with open(args.corpus, "r") as fin, open(args.outfile + "_english.txt", "w") as feng, open(args.outfile + "_fake.txt", "w") as ffake:
        for line in fin:
            tokenized = tok.tokenize(line.strip())
            feng.write(" ".join(tokenized) + "\n")
            ffake.write(" ".join([args.prefix + x for x in tokenized]) + "\n")


def main(args):
    tok = BertTokenizer.from_pretrained(args.vocab)
    with open(args.corpus, "r") as fin, open(args.outfile, "w") as feng:
        for line in tqdm(fin):
            tokenized = tok.tokenize(line.strip())
            feng.write(" ".join([args.prefix + x for x in tokenized]) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=None, type=str, required=True, help="")
    parser.add_argument("--vocab", default=None, type=str, required=True, help="")
    parser.add_argument("--prefix", default=None, type=str, required=True, help="")
    parser.add_argument("--outfile", default=None, type=str, required=True, help="")
    args = parser.parse_args()
    main(args)
