# fastflow_inn/sequence_inn.py
from typing import Iterable, Tuple, List
import torch.nn as nn
import torch
from torch import Tensor

from .invertible_module import InvertibleModule   # ← 경로만 수정


class SequenceINN(InvertibleModule):
    def __init__(self, *dims: int, force_tuple_output=False):
        super().__init__([dims])
        self.shapes = [tuple(dims)]
        self.conditions = []
        self.module_list = nn.ModuleList()
        self.force_tuple_output = force_tuple_output

    def append(self, module_class, cond=None, cond_shape=None, **kwargs):
        dims_in = [self.shapes[-1]]
        self.conditions.append(cond)
        if cond is not None:
            kwargs['dims_c'] = [cond_shape]

        if isinstance(module_class, InvertibleModule):
            module = module_class
            if module.dims_in != dims_in:
                raise ValueError(
                    f"You passed an instance of {module.__class__} to "
                    f"SequenceINN which expects {module.dims_in}, "
                    f"but previous layer outputs {dims_in}."
                )
        else:
            module = module_class(dims_in, **kwargs)

        self.module_list.append(module)
        out_dims = module.output_dims(dims_in)
        if len(out_dims) != 1:
            raise ValueError(
                f"Module {module.__class__} has >1 outputs: {out_dims}"
            )
        self.shapes.append(out_dims[0])

    # Python list 인터페이스 그대로 노출
    def __getitem__(self, i): return self.module_list[i]
    def __len__(self): return len(self.module_list)
    def __iter__(self): return iter(self.module_list)

    def output_dims(self, input_dims: List[Tuple[int]] = None):
        if input_dims is not None:
            if self.force_tuple_output:
                if input_dims != self.shapes[0]:
                    raise ValueError(
                        f"Input shapes {input_dims!r} mismatch initial "
                        f"shape {self.shapes[0]}"
                    )
            else:
                raise ValueError("Set force_tuple_output=True to query dims.")
        return [self.shapes[-1]]

    def forward(self, x_or_z: Tensor, c: Iterable[Tensor] = None,
                rev: bool = False, jac: bool = True):
        iterator = range(len(self.module_list))
        log_det_jac = 0.0
        if rev:
            iterator = reversed(iterator)

        if torch.is_tensor(x_or_z):
            x_or_z = (x_or_z,)

        for i in iterator:
            if self.conditions[i] is None:
                x_or_z, j = self.module_list[i](x_or_z, jac=jac, rev=rev)
            else:
                x_or_z, j = self.module_list[i](
                    x_or_z, c=[c[self.conditions[i]]], jac=jac, rev=rev
                )
            log_det_jac = j + log_det_jac

        return (x_or_z if self.force_tuple_output else x_or_z[0],
                log_det_jac)
