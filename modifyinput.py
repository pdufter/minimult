import torch
from typing import Set, Tuple, Text


def get_is_shifted(inputids: torch.Tensor, shift: int) -> torch.Tensor:
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
    '''
    why this should work:
    [PAD]: check if special role for masking out attention stuff?
    [UNK]: should not be used in the corpus, double check this
    [CLS]: just a token
    [SEP]: just a token
    [MASK]: just a token, actual predictions are made via the label tensor
    '''
    is_shifted = get_is_shifted(inputids, shift)
    shift_dict = {k: k + shift for k in special_token_indices}
    for k, v in shift_dict.items():
        inputids[(inputids == k) & is_shifted] = v
