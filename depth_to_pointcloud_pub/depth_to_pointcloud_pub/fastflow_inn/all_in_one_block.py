# fastflow_inn/all_in_one_block.py
import torch
import torch.nn as nn
from torch import Tensor
from .invertible_module import InvertibleModule


class AllInOneBlock(InvertibleModule):
    """
    Simplified affine-coupling + (optional) 1×1 convolution permutation.
    Compatible with FrEIA signature.
    """

    def __init__(self, dims_in, subnet_constructor,
                 affine_clamping=2.0, permute_soft=False):
        super().__init__(dims_in)
        C, H, W = dims_in[0]
        self.clamp = affine_clamping
        self.permute_soft = permute_soft

        # channel split
        self.split_len1 = C // 2
        self.split_len2 = C - self.split_len1

        # subnet: out-channels = 2*split_len2  (scale & shift)
        self.subnet = subnet_constructor(self.split_len1,
                                         2 * self.split_len2)

        if self.permute_soft:
            self.W = nn.Conv2d(C, C, kernel_size=1, bias=False)

    def output_dims(self, input_dims):
        return input_dims  # shape unchanged

    def _exp_clamped(self, s: Tensor) -> Tensor:
        # scale in log-space with clamping
        return torch.exp(torch.tanh(s) * self.clamp)

    def forward(self, x_or_z, c=None, rev=False, jac=True):
        (z,) = x_or_z  # SequenceINN wraps inputs in tuple

        if self.permute_soft and not rev:
            z = self.W(z)

        z1, z2 = torch.split(
            z, [self.split_len1, self.split_len2], dim=1)

        h = self.subnet(z1)
        s, t = torch.split(h, self.split_len2, dim=1)
        exp_s = self._exp_clamped(s)

        if not rev:
            z2 = z2 * exp_s + t
            logdet = torch.sum(torch.log(exp_s),
                               dim=[1, 2, 3])  # per-sample
        else:
            z2 = (z2 - t) / exp_s
            logdet = -torch.sum(torch.log(exp_s),
                                dim=[1, 2, 3])

        z_out = torch.cat([z1, z2], dim=1)

        if self.permute_soft and rev:
            # inverse 1×1 conv
            W_inv = torch.inverse(
                self.W.weight.squeeze().t()).t().view_as(self.W.weight)
            z_out = torch.nn.functional.conv2d(
                z_out, W_inv, bias=None)

        # SequenceINN expects (outputs, jacobian)
        if jac:
            return (z_out,), logdet
        else:
            # still return a tensor so that SequenceINN can accumulate
            return (z_out,), z_out.new_zeros(z_out.size(0))
