# fastflow_inn/invertible_module.py
# ────────────────────────────────────────────────────────────────
#  ⇢ 바로 당신이 제공한 원본을 100 % 그대로 넣었습니다.
from typing import Tuple, Iterable, List
import torch.nn as nn
from torch import Tensor

__all__ = ["InvertibleModule"]


def list_of_int_tuples(list_of_tuples):
    BASIC_ERROR = (
        f"Invalid dimension specification: You passed {list_of_tuples}, but a "
        f"list of int tuples was expected. Problem:"
    )
    try:
        iter(list_of_tuples)
    except TypeError:
        raise TypeError(f"{BASIC_ERROR} {list_of_tuples!r} cannot be iterated.")
    for int_tuple in list_of_tuples:
        try:
            iter(int_tuple)
        except TypeError:
            try:
                int(int_tuple)
                addendum = (
                    " Even if you have only one input, "
                    "you need to wrap it in a list."
                )
            except TypeError:
                addendum = ""
            raise TypeError(
                f"{BASIC_ERROR} {int_tuple!r} cannot be iterated.{addendum}"
            )
        for supposed_int in int_tuple:
            try:
                int(supposed_int)
            except TypeError:
                raise TypeError(
                    f"{BASIC_ERROR} {supposed_int!r} cannot be cast to an int."
                )
    return [tuple(map(int, int_tuple)) for int_tuple in list_of_tuples]


class InvertibleModule(nn.Module):
    def __init__(self, dims_in: Iterable[Tuple[int]],
                 dims_c: Iterable[Tuple[int]] = None):
        super().__init__()
        if dims_c is None:
            dims_c = []
        self.dims_in = list_of_int_tuples(dims_in)
        self.dims_c = list_of_int_tuples(dims_c)

    def forward(self, x_or_z: Iterable[Tensor], c: Iterable[Tensor] = None,
                rev: bool = False, jac: bool = True):
        raise NotImplementedError

    def log_jacobian(self, *args, **kwargs):
        raise DeprecationWarning("module.log_jacobian(...) is deprecated.")

    def output_dims(self, input_dims: List[Tuple[int]]):
        raise NotImplementedError
