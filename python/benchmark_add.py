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

def run_benchmark(*size):
    torch.compiler.reset()
    a = torch.randn(size)

    class IncrementModule(torch.nn.Module):
        def forward(self, a):
            return (a + 1) + 1

    results = []

    model = gen(IncrementModule(), torch_to_gern)
    # opt_M = torch.compile(IncrementModule(), backend=custom_backend)

    tiled = benchmark.Timer(
        setup=f'model(a)',
        stmt=f'model(a)',
        globals={'model': model, 'a': a},
        label=f"add {size}",
        description="add tiled",
    )

    results.append(tiled.adaptive_autorange())

    torch.compiler.reset()

    model = gen(IncrementModule(), torch_to_gern, tile_rows=size[2])

    untiled = benchmark.Timer(
        setup=f'model(a)',
        stmt=f'model(a)',
        globals={'model': model, 'a': a},
        label=f"add {size}",
        description="add untiled",
    )
    results.append(untiled.adaptive_autorange())

    torch.compiler.reset()
    model = IncrementModule()
    unoptimized = benchmark.Timer(
        setup=f'model(a)',
        stmt=f'model(a)',
        globals={'model': model, 'a': a},
        label=f"add {size}",
        description="add raw",
    )
    results.append(unoptimized.adaptive_autorange())

    torch.compiler.reset()
    model = torch.compile(IncrementModule())
    default_tc = benchmark.Timer(
        setup=f'model(a)',
        stmt=f'model(a)',
        globals={'model': model, 'a': a},
        label=f"add {size}",
        description="add default torch compile",
    )
    results.append(default_tc.adaptive_autorange())

    compare = benchmark.Compare(results)
    compare.print()
    return results

all_results = []
all_results.extend(run_benchmark(1, 1, 1024, 512))
all_results.extend(run_benchmark(1, 12, 1024, 64))
benchmark.Compare(all_results).print()