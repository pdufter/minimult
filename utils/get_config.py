import transformers
from typing import Text
import argparse


def get_config(outpath: Text) -> None:
    config = transformers.BertConfig.from_pretrained("bert-base-multilingual-cased")
    config.to_json_file(outpath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--outpath", default=None, type=str, required=True, help="")
    args = parser.parse_args()
    get_config(args.outpath)
