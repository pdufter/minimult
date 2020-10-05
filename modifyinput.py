import torch
from typing import Any, Set, Tuple, Text, Union, List
from sklearn.metrics.pairwise import cosine_distances
import numpy as np
VECMAP = None
# TMPCOUNT = -1


def get_is_shifted(inputids: Union[torch.Tensor, List], shift: int) -> Union[torch.Tensor, bool]:
    if isinstance(inputids, list):
        return not all([x < shift for x in inputids])
    else:
        not_shifted = torch.all(inputids < shift, axis=1).unsqueeze(1)
        return ~not_shifted


def invert(inputids: torch.Tensor, shift: int) -> None:
    # invert all where token id >= shift
    # this leaves out padding etc.
    to_shift = inputids >= shift
    for example, shift_entries in zip(inputids, to_shift):
        example[shift_entries] = torch.flip(example[shift_entries], (0,))


def get_language_specific_positions(inputids: torch.Tensor, shift: int, max_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
    input_shape = inputids.size()
    seq_length = input_shape[1]
    position_ids = torch.arange(seq_length, dtype=torch.long)
    position_ids = position_ids.unsqueeze(0).expand(input_shape)
    token_type_ids = torch.zeros(input_shape, dtype=torch.long)

    is_shifted = get_is_shifted(inputids, shift)
    position_ids = position_ids + is_shifted.type(dtype=torch.int) * max_length
    # assume there is only two token_type_ids
    token_type_ids = token_type_ids + is_shifted.type(dtype=torch.int) * 1
    return position_ids, token_type_ids


def shift_special_tokens(inputids: torch.Tensor, shift: int, special_token_indices: Set[int]) -> None:
    is_shifted = get_is_shifted(inputids, shift)
    shift_dict = {k: k + shift for k in special_token_indices}
    for k, v in shift_dict.items():
        inputids[(inputids == k) & is_shifted] = v


def replace_with_dict(inputids: torch.Tensor, model: Any, indices_random: torch.Tensor) -> None:
    shift = model.config.shift
    e2f = dict([(k, k + shift) for k in range(5, 200)])
    f2e = {v: k for k, v in e2f.items()}
    dictionary = {**e2f, **f2e}
    for k, v in dictionary.items():
        inputids[(inputids == k) & indices_random] = v


def replace_with_nn(inputids: torch.Tensor, model: Any, indices_random: torch.Tensor, replace_with_nn: int, vecmap_space: Text = None, tok: Any = None) -> None:
    large_number = 999
    shift = model.config.shift
    if vecmap_space is not None:
        if VECMAP is None:
            with open(vecmap_space, "r") as fin:
                tmp = {}
                dim = 0
                for line in fin:
                    word, vector = line.split()[0], line.split()[1:]
                    tmp[word] = [float(x) for x in vector]
                    dim = len(vector)
                embeddings = []
                for word, index in tok.vocab.items():
                    embeddings.append(tmp.get(word, np.zeros(dim)))
                embeddings = np.array(embeddings)
        else:
            embeddings = VECMAP
    else:
        embeddings = model.bert.embeddings.word_embeddings.weight.detach().cpu().numpy()
    queries = inputids[indices_random]
    if len(queries) == 0:
        return
    is_shifted = get_is_shifted(queries.unsqueeze(1), shift).squeeze()
    dist = cosine_distances(np.take(embeddings, queries, axis=0), embeddings)
    # restrict nn search to the other language
    large = np.zeros_like(dist)
    large[np.array(is_shifted), shift:] = large_number
    large[~np.array(is_shifted), :shift] = large_number
    dist += large
    nns = torch.LongTensor(np.argsort(dist, axis=1)[:, :replace_with_nn])
    choice = torch.randint(low=0, high=nns.shape[1], size=(nns.shape[0], 1))
    inputids[indices_random] = torch.gather(nns, 1, choice).squeeze()

