import torch
import torch.fx
import torch.utils.benchmark as benchmark
import operator
from generate_torch_compile_pipeline_4d import gen, FnInterface
import math
from gern_py import *

torch_to_gern = {
    operator.add: FnInterface(MatrixAddCPU4D)
}

def check(*size):
    torch.compiler.reset()
    a = torch.randn(size)

    class IncrementModule(torch.nn.Module):
        def forward(self, a):
            return (a + 1) + 1

    opt_M = gen(IncrementModule(), torch_to_gern)
    assert(torch.allclose(opt_M(a), IncrementModule()(a)))

    torch.compiler.reset()

    opt_M = gen(IncrementModule(), torch_to_gern, tile_rows=size[2])
    assert(torch.allclose(opt_M(a), IncrementModule()(a)))

check(1, 1, 1024, 512)
check(1, 12, 1024, 64)