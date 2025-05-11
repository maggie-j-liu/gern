import torch
import google_benchmark as benchmark
import operator
from generate_torch_compile_pipeline_4d import gen, FnInterface
from gern_py import *
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

torch_to_gern = {
    operator.add: FnInterface(MatrixAddCPU4D)
}

class IncrementModule(torch.nn.Module):
    def forward(self, a):
        return torch.add(torch.add(a, 1), 1)

# @benchmark.register
@benchmark.option.repetitions(5)
@benchmark.option.min_time(4)
@benchmark.option.min_warmup_time(4)
@benchmark.option.args([1, 1, 1024, 512])
@benchmark.option.args([1, 12, 1024, 64])
@benchmark.option.display_aggregates_only(True)
def tiled_add(state):
    torch.compiler.reset()
    model = gen(IncrementModule(), torch_to_gern)
    a = torch.randn((state.range(0), state.range(1), state.range(2), state.range(3))) 
    model(a)
    a = torch.randn((state.range(0), state.range(1), state.range(2), state.range(3))) 
    while state:
        model(a)

# @benchmark.register
@benchmark.option.repetitions(5)
@benchmark.option.min_time(4)
@benchmark.option.min_warmup_time(4)
@benchmark.option.args([1, 1, 1024, 512])
@benchmark.option.args([1, 12, 1024, 64])
@benchmark.option.display_aggregates_only(True)
def untiled_add(state):
    torch.compiler.reset()
    model = gen(IncrementModule(), torch_to_gern, tile_rows=state.range(2))
    a = torch.randn((state.range(0), state.range(1), state.range(2), state.range(3))) 
    model(a)
    a = torch.randn((state.range(0), state.range(1), state.range(2), state.range(3))) 
    while state:
        model(a)

@benchmark.register
@benchmark.option.repetitions(5)
@benchmark.option.min_time(4)
@benchmark.option.min_warmup_time(4)
@benchmark.option.args([1, 1, 1024, 512])
# @benchmark.option.display_aggregates_only(True)
def torch_add(state):
    a = torch.randn((state.range(0), state.range(1), state.range(2), state.range(3)), dtype=torch.float, device="cpu") 
    b = torch.empty((state.range(0), state.range(1), state.range(2), state.range(3)), dtype=torch.float, device="cpu")
    while state:
        torch.add(a, 1, out=b)
        

@benchmark.register
@benchmark.option.repetitions(5)
@benchmark.option.min_time(4)
@benchmark.option.min_warmup_time(4)
@benchmark.option.args([1, 1, 1024, 512])
# @benchmark.option.display_aggregates_only(True)
def add_in_place(state):
    a = torch.randn((state.range(0), state.range(1), state.range(2), state.range(3)), dtype=torch.float, device="cpu") 
    while state:
        a.add_(1)

@benchmark.register
@benchmark.option.repetitions(5)
@benchmark.option.min_time(4)
@benchmark.option.min_warmup_time(4)
@benchmark.option.args([1, 1, 1024, 512, 1, 1, 512, 1024])
# @benchmark.option.display_aggregates_only(True)
def matmul(state):
    a = torch.randn((state.range(0), state.range(1), state.range(2), state.range(3)), dtype=torch.float, device="cpu")
    b = torch.randn((state.range(4), state.range(5), state.range(6), state.range(7)), dtype=torch.float, device="cpu")
    c = torch.empty((state.range(0), state.range(1), state.range(2), state.range(7)), dtype=torch.float, device="cpu")
    while state:
        torch.matmul(a, b, out=c) 
        

if __name__ == "__main__":
    benchmark.main()