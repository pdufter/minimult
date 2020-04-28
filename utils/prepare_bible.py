from projects.posanalysis.data import PBC
from nltk import sent_tokenize
from typing import Text
import argparse


def get_simple_bible(basepath: Text, edition: Text, outpath: Text, clean_punctuation: bool) -> None:
    p = PBC(basepath)
    data = p.load_single_edition(edition)

    with open(outpath, "w") as fout:
        for k, v in data.items():
            if clean_punctuation:
                v = v.replace("”", "")
                v = v.replace("’", "")
                v = v.replace("‘", "")
                v = v.replace("“", "")
                v = v.replace("(", "")
                v = v.replace(")", "")
                # v = v.replace(" :", ":")
            # get rid of double spaces
            v = " ".join(v.split())
            for sentence in sent_tokenize(v):
                fout.write(sentence + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pbc", default="/nfs/datc/pbc/", type=str, required=False, help="")
    parser.add_argument("--edition", default="eng_easytoread", type=str, required=True, help="")
    parser.add_argument("--outpath", default="/mounts/work/philipp/multl/corpora/bible_eng_easytoread.txt",
                        type=str, required=True, help="")
    parser.add_argument("--clean_punctuation", action="store_true", help="")
    args = parser.parse_args()

    get_simple_bible(args.pbc, args.edition, args.outpath, args.clean_punctuation)
