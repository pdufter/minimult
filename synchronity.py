from simalign import SentenceAligner
from typing import Text, Dict, Callable, Any, List, Iterable, Tuple
import os
from tqdm import tqdm
import argparse
from transformers import BertModel, BertTokenizer
import numpy as np
from nltk.metrics.distance import edit_distance
import collections


def rec_dd():
    return collections.defaultdict(rec_dd)


def load_embedding_model(bert_model):
    embedding_model = BertModel.from_pretrained(bert_model, output_hidden_states=True)
    embedding_model.eval()
    embedding_model.to('cuda')
    embedding_tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=False)
    return embedding_model, embedding_tokenizer


class Corpus(object):
    """docstring for Corpus"""

    def __init__(self) -> None:
        super(Corpus, self).__init__()
        # Format: Language, SubID, Sentence ID, Text
        self.text = rec_dd()

    @staticmethod
    def apply_function(corpusview: Dict,
                       modification: Callable[[Any], Any]) -> Dict:
        new_view = rec_dd()
        for lang in corpusview:
            for subid in corpusview[lang]:
                for sentenceid in corpusview[lang][subid]:
                    new_view[lang][subid][sentenceid] = modification(
                        corpusview[lang][subid][sentenceid])
        return new_view

    def tokenize(self, tokenize_function: Callable[[Text], List[Text]]) -> None:
        self.tokenized_text = self.apply_function(self.text, tokenize_function)

    def get_ids(self, id_function: Callable[[List[Text]], List[int]]) -> None:
        self.ids = self.apply_function(self.tokenized_text, id_function)

    def get_all_sentence_ids(self) -> List[Any]:
        all_ids = []
        for lang in self.text:
            for subid in self.text[lang]:
                all_ids.extend(list(self.text[lang][subid].keys()))
        return all_ids

    def keep_first_n_sentences(self, n: int) -> None:
        views = [self.text, self.tokenized_text, self.ids]
        for view in views:
            for lang in view:
                for subid in view[lang]:
                    for i, sentenceid in enumerate(list(view[lang][subid].keys())):
                        if i >= n:
                            del view[lang][subid][sentenceid]

    def keep_sentences(self, sentenceids: Iterable) -> None:
        sentenceids = set(sentenceids)
        views = [self.text, self.tokenized_text, self.ids]
        for view in views:
            for lang in view:
                for subid in view[lang]:
                    for i, sentenceid in enumerate(list(view[lang][subid].keys())):
                        if sentenceid not in sentenceids:
                            del view[lang][subid][sentenceid]


class XNLI(Corpus):
    def __init__(self) -> None:
        super(XNLI, self).__init__()
        self.langs = ["ar", "de", "en", "fr", "ru", "th", "ur", "zh", "bg", "el", "es", "hi", "sw", "tr", "vi"]

    def load_train(self, path: Text) -> None:
        for lang in tqdm(self.langs):
            with open(os.path.join(path, "multinli.train.{}.tsv".format(lang)), "r") as fp:
                next(fp)
                for i, line in enumerate(fp):
                    line = line.strip()
                    prem, hypo, _ = line.split("\t")
                    self.text[lang]["train"][2 * i] = prem.strip()
                    self.text[lang]["train"][2 * i + 1] = hypo.strip()

    def load_test(self, path: Text, subid: Text) -> None:
        with open(path, "r") as fp:
            next(fp)
            for i, line in enumerate(fp):
                line = line.strip()
                (language,
                 gold_label,
                 sentence1_binary_parse,
                 sentence2_binary_parse,
                 sentence1_parse,
                 sentence2_parse,
                 sentence1,
                 sentence2,
                 promptID,
                 pairID,
                 genre,
                 label1,
                 label2,
                 label3,
                 label4,
                 label5,
                 sentence1_tokenized,
                 sentence2_tokenized,
                 match) = line.split("\t")
                self.text[language][subid][2 * int(pairID)] = sentence1.strip()
                self.text[language][subid][2 * int(pairID) + 1] = sentence2.strip()


def remove_null_alignments(alignment: np.ndarray) -> np.ndarray:
    n, m = alignment.shape
    keep0 = alignment.sum(axis=1) > 0
    keep1 = alignment.sum(axis=0) > 0
    if sum(~keep0) > 0 and sum(~keep1) > 0:
        raise ValueError("Matching not maximal.")
    alignment = alignment[keep0, :]
    alignment = alignment[:, keep1]
    return alignment


def kendallstaudistance(pi: List[int], sigma: List[int]) -> float:
    assert len(pi) == len(sigma), "Length Mismatch."
    n = len(pi)
    z = 0
    for i in range(n):
        for j in range(n):
            if pi[i] < pi[j] and sigma[i] > sigma[j]:
                z += 1
            else:
                z += 0
    return 1 - np.sqrt(2 * z / (n**2 - n))


def hammingdistance(pi: List[int], sigma: List[int]) -> float:
    assert len(pi) == len(sigma), "Length Mismatch."
    n = len(pi)
    x = sum([a == b for a, b in zip(pi, sigma)])
    return 1 - x / n


def brevity_penalty(n: int, m: int) -> float:
    if m > n:
        penalty = 1
    else:
        penalty = np.exp(1 - n / m)
    return penalty


def list2matrix(alignment: List[Tuple[int, int]], n: int, m: int) -> np.ndarray:
    matrix = np.zeros((n, m))
    matrix[tuple(zip(*alignment))] = 1
    return matrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--xnli_dir", default="/mounts/work/philipp/data/xnli/XNLI/", type=str, help="")
    parser.add_argument("--language1", default="en", type=str, help="")
    parser.add_argument("--language2", default="", type=str, help="")
    parser.add_argument("--maxn", default=500, type=int, help="")
    parser.add_argument("--model", default="bert-base-multilingual-cased", type=str, help="")
    parser.add_argument("--distance", default="kendall", type=str, help="")
    parser.add_argument("--outfile", default="", type=str, help="")
    args = parser.parse_args()
    # load
    xnli = XNLI()
    xnli.load_test(args.xnli_dir, "dev")

    _, tok = load_embedding_model(args.model)
    myaligner = SentenceAligner(model="bert", token_type="bpe", matching_methods="m")
    all_scores = {}
    if args.language2 == "":
        args.language2 = ",".join(xnli.langs)
    for lang2 in args.language2.split(","):
        # align
        sentences_1 = xnli.text[args.language1]["dev"]
        sentences_2 = xnli.text[lang2]["dev"]
        ids1 = list(sentences_1.keys())
        ids2 = list(sentences_2.keys())
        if len(set(ids1) ^ set(ids2)) > 0:
            raise ValueError("Data not parallel?")
        all_kendall = []
        all_penalty = []
        for _, sid in tqdm(zip(range(args.maxn), set(ids1) & set(ids2))):
            # align
            e = tok.tokenize(sentences_1[sid])
            f = tok.tokenize(sentences_2[sid])
            if min(len(e), len(f)) in [0, 1]:
                print("emtpy sentences: ", lang2)
                print("emtpy sentences: ", sentences_1[sid], sentences_2[sid])
                print("emtpy sentences: ", e, f)
                continue
            alignment = myaligner.get_word_aligns(e, f)
            alignment = list2matrix(alignment['mwmf'], len(e), len(f))
            # postprocess
            alignment = remove_null_alignments(alignment)
            # compute score
            permutation = alignment.argmax(axis=1)
            # todo: what about axis=0?
            gold_permutation = np.arange(len(permutation))
            if args.distance == "kendall":
                kendalls = kendallstaudistance(gold_permutation, permutation)
            elif args.distance == "hamming":
                kendalls = hammingdistance(gold_permutation, permutation)
            elif args.distance == "leven":
                kendalls = edit_distance("".join([str(x) for x in gold_permutation]),
                                         "".join([str(x) for x in permutation]), transpositions=True)
            penalty = brevity_penalty(len(e), len(f))
            # print("{:.2f} - {:.2f}".format(kendalls, penalty))
            all_kendall.append(kendalls)
            all_penalty.append(penalty)
        all_kendall = np.array(all_kendall)
        all_penalty = np.array(all_penalty)
        overall = all_kendall * all_penalty
        all_scores["en-{}".format(lang2)] = overall, all_kendall, all_penalty
        # print("Overall score is {}".format(overall_score))

    np.save(args.outfile + ".npy", all_scores)
    with open(args.outfile + ".txt", "w") as fp:
        fp.write("MEAN---\n")
        for lpair, (overall, all_kendall, all_penalty) in all_scores.items():
            fp.write("{} {:.2f} {:.2f} {:.2f}\n".format(lpair, overall.mean(), all_kendall.mean(), all_penalty.mean()))
        fp.write("MIN---\n")
        for lpair, (overall, all_kendall, all_penalty) in all_scores.items():
            fp.write("{} {:.2f} {:.2f} {:.2f}\n".format(lpair, overall.min(), all_kendall.min(), all_penalty.min()))
        fp.write("STD---\n")
        for lpair, (overall, all_kendall, all_penalty) in all_scores.items():
            fp.write("{} {:.2f} {:.2f} {:.2f}\n".format(lpair, overall.std(), all_kendall.std(), all_penalty.std()))
        fp.write("MEDIAN---\n")
        for lpair, (overall, all_kendall, all_penalty) in all_scores.items():
            fp.write("{} {:.2f} {:.2f} {:.2f}\n".format(lpair, np.percentile(overall, 50),
                                                        np.percentile(all_kendall, 50), np.percentile(all_penalty, 50)))
    '''
    all_scores = np.load(infile).tolist()
    xnli_perf = {"en-ar": 64.90,
                 "en-de": 71.10,
                 "en-en": 81.40,
                 "en-fr": 73.80,
                 "en-ru": 69.00,
                 "en-th": 55.80,
                 "en-ur": 58.00,
                 "en-zh": 69.30,
                 "en-bg": 68.90,
                 "en-el": 66.40,
                 "en-es": 74.30,
                 "en-hi": 60.00,
                 "en-sw": 50.40,
                 "en-tr": 61.60,
                 "en-vi": 69.50}

    from scipy.stats import pearsonr
    lpairs = list(xnli_perf.keys())

    print(pearsonr([xnli_perf[lpair] for lpair in lpairs], [all_scores[lpair][1].mean() for lpair in lpairs]))
    '''
