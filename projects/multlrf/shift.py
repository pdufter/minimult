from typing import List, Text, Set


def shift_example(original: List[int], do_not_shift: Set[int], shift: int):
    tmp = []
    for x in original:
        if x in do_not_shift:
            tmp.append(x)
        else:
            tmp.append(x + shift)
    return tmp


def add_shifted_input(original: List[List[int]], do_not_shift: Set[int], shift: int) -> None:
    to_add = []
    for example in original:
        to_add.append(shift_example(example, do_not_shift, shift))
    original.extend(to_add)


def remove_parallel_data(original: List[List[int]]) -> None:
    if len(original) % 4 != 0:
        raise ValueError("Data not parallel at all?")
    quarter = int(len(original) / 4)
    modified = original[:quarter] + original[-quarter:]
    return modified
