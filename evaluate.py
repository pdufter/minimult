from projects.multlrf.utils import utils
from projects.multlrf.run_language_modeling import load_and_cache_examples, mask_tokens, LineByLineTextDataset
from torch.nn.utils.rnn import pad_sequence
from projects.multlrf.shift import add_shifted_input, remove_parallel_data
from projects.multlrf.modifymodel import delete_position_segment_embeddings
from projects.multlrf.modifyinput import invert, get_language_specific_positions, shift_special_tokens
import logging
import argparse
from typing import List, Any, Set, Tuple, Text
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from tqdm import tqdm
import collections
from sklearn.metrics.pairwise import cosine_distances
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


logger = logging.getLogger(__name__)


def remove_padding(tensor, padding_mask):
    if tensor is None:
        return []
    result = []
    for elem, mask in zip(tensor, padding_mask):
        if len(elem[mask]) == 0 and len(elem) > 2:
            # add the first token after [CLS] if entry is too short
            result.append(elem[1:2])
        else:
            result.append(elem[mask])
    return result


def load_and_preprocess_corpus(eval_dataset, add_fake_english, args, tokenizer, model):
    if add_fake_english:
        add_shifted_input(eval_dataset.examples, args.special_token_indices, model.config.shift)

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate
    )

    model.eval()
    mycounter = 0
    all_vectors = collections.defaultdict(list)
    all_inputs = collections.defaultdict(list)
    count = -1
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        count += 1
        if count == 8:
            pass
        if args.invert_order:
            invert(batch, model.config.shift)
        if args.language_specific_positions:
            if args.block_size > 256:
                raise ValueError("Language specific posiiton embeddings can only be <256.")
            position_ids, segment_ids = get_language_specific_positions(batch, model.config.shift, args.block_size)
            position_ids = position_ids.to(args.device)
            segment_ids = segment_ids.to(args.device)
        else:
            position_ids, segment_ids = None, None
        # we do not require masking
        # inputs, labels = mask_tokens(batch, tokenizer, args) if args.mlm else (batch, batch)
        if args.shift_special_tokens:
            shift_special_tokens(batch, model.config.shift, args.special_token_indices)
        # directly get vectors
        inputs = batch.to(args.device)

        with torch.no_grad():
            outputs = model(inputs, position_ids=position_ids, token_type_ids=segment_ids)
            padding_mask = (inputs == tokenizer.pad_token_id) | (inputs == tokenizer.pad_token_id + model.config.shift)
            padding_mask = padding_mask | (inputs == tokenizer.cls_token_id) | (inputs == tokenizer.cls_token_id + model.config.shift)
            padding_mask = padding_mask | (inputs == tokenizer.sep_token_id) | (inputs == tokenizer.sep_token_id + model.config.shift)
            padding_mask = ~padding_mask
            for layer in args.eval_layers:
                all_vectors[layer].extend(remove_padding(outputs[1][layer], padding_mask))
            all_inputs["input_ids"].extend(remove_padding(inputs, padding_mask))
            all_inputs["position_ids"].extend(remove_padding(position_ids, padding_mask))
            all_inputs["token_type_ids"].extend(remove_padding(segment_ids, padding_mask))
    # for k, v in all_vectors.items():
    #     all_vectors[k] = torch.cat(all_vectors[k], dim=0)
    return all_vectors, all_inputs


class VocabDataset(LineByLineTextDataset):
    def __init__(self, tokenizer: Any, args, block_size=512):
        self.examples = [[2, i, 3] for i in range(len(tokenizer.vocab))]


def get_distances(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return cosine_distances(X, Y)


def get_alignment_matrix(dist: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    m, n = dist.shape
    forward = np.eye(n)[dist.argmin(axis=1)]  # m x n
    backward = np.eye(m)[dist.argmin(axis=0)]  # n x m
    return forward, backward.transpose()


def symmetrize(forward: np.ndarray, backward: np.ndarray) -> np.ndarray:
    return forward * backward


def evaluate_retrieval(vectors, args):
    # for each sentence pair get vectors and alignments (using NNs)
    n = len(vectors)
    if n % 2 != 0:
        raise ValueError("Something's wrong.")
    vectors_e, vectors_f = vectors[:n // 2], vectors[n // 2:]
    vectors_e = np.array([x.cpu().numpy().mean(axis=0) for x in vectors_e])
    vectors_f = np.array([x.cpu().numpy().mean(axis=0) for x in vectors_f])
    dist = get_distances(vectors_e, vectors_f)
    if dist.shape[0] != dist.shape[1]:
        print("Number of sentences is different?")
    # get different p@k
    nns = np.argsort(dist, axis=1)[:, :10]
    gt = np.arange(dist.shape[0]).reshape(-1, 1)
    p = {}
    for considern in [1, 5, 10]:
        hits1 = ((nns[:, :considern] == gt).sum(axis=1) > 0).sum()
        p[considern] = hits1 / dist.shape[0]
    nns = np.argsort(dist, axis=0)[:10, :].transpose()
    gt = np.arange(dist.shape[0]).reshape(-1, 1)
    pinv = {}
    for considern in [1, 5, 10]:
        hits1 = ((nns[:, :considern] == gt).sum(axis=1) > 0).sum()
        pinv[considern] = hits1 / dist.shape[0]
    return p, pinv


def evaluate_alignment(vectors, args):
    # for each sentence pair get vectors and alignments (using NNs)
    n = len(vectors)
    if n % 2 != 0:
        raise ValueError("Somethings wrong.")
    vectors_e, vectors_f = vectors[:n // 2], vectors[n // 2:]
    all_predict = collections.defaultdict(list)
    all_trues = []
    for i, (e, f) in tqdm(enumerate(zip(vectors_e, vectors_f)), desc='Alignment'):
        if min(min(e.shape), min(f.shape)) == 0:
            raise ValueError("Empty sentence.")
        dist = get_distances(e.cpu().numpy(), f.cpu().numpy())
        forward, backward = get_alignment_matrix(dist)
        intersect = symmetrize(forward, backward)
        if dist.shape[0] != dist.shape[1]:
            raise ValueError("Sentence length is different: {}.".format(i))
        gold = np.eye(dist.shape[0])
        if args.invert_order:
            gold = np.flip(gold, (1,))
        # filter edges where all systems predicted 0
        non_zero = (forward.flatten() + backward.flatten() + intersect.flatten() + gold.flatten()) > 0
        all_predict["forward"].extend(list(forward.flatten()[non_zero]))
        all_predict["backward"].extend(list(backward.flatten()[non_zero]))
        all_predict["intersect"].extend(list(intersect.flatten()[non_zero]))
        all_trues.extend(list(gold.flatten()[non_zero]))

    result = {}
    for k, v in all_predict.items():
        acc = accuracy_score(all_trues, v)
        prec = precision_score(all_trues, v)
        rec = recall_score(all_trues, v)
        f1 = f1_score(all_trues, v)
        result[k] = acc, prec, rec, f1

    return result


def load_perplexity(args):
    with open(os.path.join(args.model_name_or_path, "eval_results.txt"), "r") as fp:
        text = fp.read().strip()
        text = text.replace("perplexity = tensor(", "")
        text = text.replace(")", "")
        perplexity = float(text)
    return perplexity


def get_modifications(args, config):
    modifications = []
    if args.language_specific_positions:
        modifications.append("lang-pos")
    if args.invert_order:
        modifications.append("inv-order")
    if args.shift_special_tokens:
        modifications.append("shift-special")
    if args.replacement_probs:
        modifications.append("repl-probs({},{})".format(*args.replacement_probs.split(",")))
    if args.delete_position_segment_embeddings:
        modifications.append("del-pos")
    if args.do_not_replace_with_random_words:
        modifications.append("no-random")
    if args.no_parallel_data:
        modifications.append("no-parallel")
    if config.hidden_size > 64:
        modifications.append("overparam")
    if len(modifications) == 0:
        modifications.append("original")
    return ";".join(modifications)


def format_result_line(args, perplexity, alignment_results, retrieval_results, translation_results, layer, config):
    modifications = get_modifications(args, config)
    formatting = "{} " * 32 + "\n"
    return formatting.format(
        args.modeltype,
        args.exid,
        args.seed,
        modifications,
        args.model_name_or_path,
        args.take_n_sentences,
        layer,
        perplexity,
        alignment_results['intersect'][3],
        0.5 * (retrieval_results[0][1] + retrieval_results[1][1]),
        0.5 * (translation_results[0][1] + translation_results[1][1]),
        alignment_results['forward'][1],
        alignment_results['forward'][2],
        alignment_results['forward'][3],
        alignment_results['backward'][1],
        alignment_results['backward'][2],
        alignment_results['backward'][3],
        alignment_results['intersect'][1],
        alignment_results['intersect'][2],
        alignment_results['intersect'][3],
        retrieval_results[0][1],
        retrieval_results[0][5],
        retrieval_results[0][10],
        retrieval_results[1][1],
        retrieval_results[1][5],
        retrieval_results[1][10],
        translation_results[0][1],
        translation_results[0][5],
        translation_results[0][10],
        translation_results[1][1],
        translation_results[1][5],
        translation_results[1][10])


def evaluate_all(args):
    outfile = open(args.outfile, "a")
    model, tokenizer, config = utils.load_embedding_model(args.model_name_or_path)
    args.special_token_indices = set([tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.mask_token_id,
        tokenizer.unk_token_id,
        tokenizer.pad_token_id])
    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)
    if args.take_n_sentences > 0:
        eval_dataset.examples = eval_dataset.examples[:args.take_n_sentences]
    vectors, inputs = load_and_preprocess_corpus(eval_dataset, args.add_fake_english, args, tokenizer, model)

    vocab_dataset = VocabDataset(tokenizer, args)
    vectors_vocab, inputs_vocab = load_and_preprocess_corpus(vocab_dataset, False, args, tokenizer, model)

    for layer in args.eval_layers:
        alignment_results = evaluate_alignment(vectors[layer], args)
        retrieval_results = evaluate_retrieval(vectors[layer], args)
        translation_results = evaluate_retrieval(vectors_vocab[layer], args)
        perplexity = load_perplexity(args)
        line = format_result_line(args, perplexity, alignment_results, retrieval_results, translation_results, layer, config)
        outfile.write(line)
    outfile.close()


def main():
    '''
    TODO
    add capability to use only n sentences
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_data_file",
        default=None,
        type=str,
        help="",
    )

    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="The model checkpoint for weights initialization.",
    )
    parser.add_argument(
        "--block_size",
        default=128,
        type=int,
        help="Optional input sequence length after tokenization."
        "The training dataset will be truncated in block of this size for training."
        "Default to the model max input length for single sentence inputs (take into account special tokens).",
    )
    parser.add_argument(
        "--eval_batch_size", default=256, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    parser.add_argument("--eval_layers", default="0,8", type=str, help="")
    parser.add_argument("--language_specific_positions", action="store_true", help="")
    parser.add_argument("--invert_order", action="store_true", help="")
    parser.add_argument("--shift_special_tokens", action="store_true", help="")
    parser.add_argument("--outfile", default="", type=str, help="", required=True)
    parser.add_argument("--exid", default="", type=str, help="", required=True)
    parser.add_argument("--seed", default=None, type=int, help="", required=True)
    parser.add_argument("--modeltype", default="", type=str, help="", required=True)
    parser.add_argument("--take_n_sentences", default=-1, type=int, help="", required=False)
    parser.add_argument("--replacement_probs", default="", type=str, help="FAKE NOT REQUIRED", required=False)
    parser.add_argument("--delete_position_segment_embeddings", action="store_true", help="FAKE NOT REQUIRED")
    parser.add_argument("--do_not_replace_with_random_words", action="store_true", help="FAKE NOT REQUIRED")
    parser.add_argument("--no_parallel_data", action="store_true", help="FAKE NOT REQUIRED")

    args = parser.parse_args()

    # add some default values
    args.line_by_line = True
    args.add_fake_english = True
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    args.eval_layers = [int(x) for x in args.eval_layers.split(",")]

    evaluate_all(args)


if __name__ == '__main__':
    main()
