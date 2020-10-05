import copy
import torch
from typing import List, Any, Text
import numpy as np
from sklearn.metrics.pairwise import cosine_distances

VECMAP = None


def shift_dataset(examples, shift):
    new_examples = []
    for example in examples:
        new_examples.append([x + shift for x in example])
    return new_examples


def merge_datasets(datasets, shifts):
    for i in range(1, len(datasets)):
        datasets[0].examples.extend(shift_dataset(datasets[i].examples, shifts[i]))
    return datasets[0]


def merge_tokenizers(tokenizers, args):
    tokenizer = copy.deepcopy(tokenizers[0])
    shifts = [0] + [len(tok.vocab) for tok in tokenizers]
    shifts = [shifts[i] + sum(shifts[:i]) for i in range(len(shifts))]
    # add vocab from other languages to the main tokenizer
    for i in range(1, len(tokenizers)):
        # prefix
        to_add = ["0" + args.language_id[i] + "0" + x[0] for x in tokenizers[i].vocab.items()]
        tokenizer.add_tokens(to_add)
    return tokenizer, shifts


def get_special_tokens_mask(batch, special_token_indices, shifts):
    mask = torch.zeros_like(batch)
    for special_token_index in special_token_indices:
        for shift in shifts[:-1]:
            mask += (batch == (special_token_index + shift))
    return mask.bool()


def get_lang_id(batch, langid, shifts, cls_token_id):
    return (batch[:, 0] == cls_token_id + shifts[langid]).unsqueeze(1)


def get_language_specific_positions(batch, shifts, max_length, tokenizer):
    input_shape = batch.size()
    seq_length = input_shape[1]
    position_ids = torch.arange(seq_length, dtype=torch.long)
    position_ids = position_ids.unsqueeze(0).repeat(input_shape[0], 1)
    segment_ids = torch.zeros_like(batch)

    for langid in range(len(shifts) - 1):
        language_mask = get_lang_id(batch, langid, shifts, tokenizer.cls_token_id)
        position_ids[language_mask.squeeze(), :] += max_length * langid
        segment_ids[language_mask.squeeze(), :] += langid
    return position_ids, segment_ids


def invert(batch, shifts, args, tokenizer):
    lang_indices_to_invert = [args.language_id.index(x) for x in args.invert_langs]
    for langid in lang_indices_to_invert:
        language_mask = get_lang_id(batch, langid, shifts, tokenizer.cls_token_id)
        special_token_mask = get_special_tokens_mask(batch, args.special_token_indices, shifts)
        for i, example in enumerate(batch):
            if language_mask[i]:
                example[~special_token_mask[i]] = torch.flip(example[~special_token_mask[i]], (0,))            


def invert_pairs(batch, shifts, args, tokenizer):
    lang_indices_to_invert = [args.language_id.index(x) for x in args.invert_langs]
    for langid in lang_indices_to_invert:
        language_mask = get_lang_id(batch, langid, shifts, tokenizer.cls_token_id)
        for i, example in enumerate(batch):
            if language_mask[i]:
                start = 1
                seps = (example == tokenizer.sep_token_id + shifts[langid]).nonzero()
                example[start:seps[0,0]] = torch.flip(example[start:seps[0,0]], (0,))
                example[seps[0,0] + 1:seps[1,0]] = torch.flip(example[seps[0,0] + 1:seps[1,0]], (0,))


def shift_special_tokens(batch, shifts, special_token_indices, tokenizer):
    for langid in range(len(shifts) - 1):
        shift_dict = {k: k + shifts[langid] for k in special_token_indices}
        language_mask = get_lang_id(batch, langid, shifts, tokenizer.cls_token_id)
        for k, v in shift_dict.items():
            batch[(batch == k) & language_mask] = v


def unshift_special_tokens(batch, shifts, special_token_indices, tokenizer):
    for langid in range(len(shifts) - 1):
        shift_dict = {k + shifts[langid]: k for k in special_token_indices}
        language_mask = get_lang_id(batch, langid, shifts, tokenizer.cls_token_id)
        for k, v in shift_dict.items():
            batch[(batch == k) & language_mask] = v


def replace_with_nn(inputids: torch.Tensor, model: Any, indices_random: torch.Tensor, replace_with_nn: int, shifts: List[int], vecmap_space: Text = None, tok: Any = None) -> None:
    global VECMAP
    if vecmap_space is not None:
        if VECMAP is None:
            with open(vecmap_space, "r") as fin:
                tmp = {}
                dim = 0
                for line in fin:
                    word, vector = line.split()[0], line.split()[1:]
                    if word.startswith("0en0"):
                        word = word.replace("0en0", "", 1)
                    tmp[word] = [float(x) for x in vector]
                    dim = len(vector)
                embeddings = []
                for word, index in tok.get_vocab().items():
                    embeddings.append(tmp.get(word, np.zeros(dim)))
                embeddings = np.array(embeddings)
                VECMAP = embeddings
        else:
            embeddings = VECMAP
    else:
        embeddings = model.bert.embeddings.word_embeddings.weight.detach().cpu().numpy()
    for langid in range(len(shifts[:1])):
        langids = get_lang_id(inputids, langid, shifts, tok.cls_token_id).repeat(1, inputids.shape[1])
        queries = inputids[indices_random & langids]
        if len(queries) == 0:
            return
        # just choose one other language
        other_lang = np.random.choice(list(set(range(len(shifts[:-1]))) - set([langid])))
        rel_embeddings = embeddings[shifts[other_lang]:shifts[other_lang + 1], :]
        dist = cosine_distances(np.take(embeddings, queries, axis=0), rel_embeddings)
        nns = torch.LongTensor(np.argsort(dist, axis=1)[:, :replace_with_nn])
        choice = torch.randint(low=0, high=nns.shape[1], size=(nns.shape[0], 1))
        inputids[indices_random & langids] = torch.gather(nns, 1, choice).squeeze() + shifts[other_lang]
