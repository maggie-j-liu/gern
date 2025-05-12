# Python Integration for Gern

Python integration for Gern consists of two parts -- Python bindings that allow for the creation of Gern programs in Python, and a custom torch.compile backend that automatically uses Gern to optimize Pytorch code.

## Python Bindings

The Python bindings are located in `src/python_bindings/bindings.cpp`. They are written with the pybind library, and expose Gern classes and methods.

In addition to Gern classes like `gern::Annotation`, `gern::Variable`, and `gern::AbstractDataType`, each abstract data type class also needs to be exposed with a pybind binding, like:

```cpp
py::class_<annot::MatrixCPU, AbstractDataType>(m, "AnnotMatrixCPU")
		.def("init", [](std::string &name){
			return new const annot::MatrixCPU(name);
		}, py::return_value_policy::reference);
```

Each annotation class also needs its own binding, for example, for the `annot:MatrixAddCPU` class. When new annotations are added, a binding must also be created to expose the annotation to Python.

```cpp
py::class_<annot::MatrixAddCPU, AbstractFunction>(m, "MatrixAddCPU")
		.def(py::init<>());
```

The implementation of the Matrix class must also be exposed with a binding, for example:

```cpp
py::class_<impl::MatrixCPU>(m, "MatrixCPU")
    .def("init", [](int64_t row, int64_t col, int64_t lda){
			auto mat = new impl::MatrixCPU(row, col, lda);
			return mat;
		}, py::return_value_policy::reference)
// more methods
```

The special `init` method is required because we need to pass in the `py::return_value_policy::reference` argument, which will ensure that C++ will manage the matrix object's lifetime, to avoid segfaults if Python tries to deallocate a Matrix that is still being used by Gern.

In addition, there are two wrapper classes (`MyInt` and `MyFloat`) that wrap int values and float values, respectively. This is because `gern::Runner::evaluate` expects a map of strings to pointers, and there's no good way to get the address from a python value. So `MyInt` and `MyFloat` will create the pointer, and the `getAddress()` function (which is also implemented for the `MatrixCPU` implementation class) will get a pointer to the value, to pass to `Runner::evaluate`.

## `torch.compile` Backend

The custom `torch.compile` backend library is implemented in `python/generate_torch_compile_pipeline.py`. It currently only handles a limited set of operations on 2D tensors, with operations on higher dimensional tensors made possible by running a function on a 2D slice, then stacking all the slices back together.

It defines

- a `FnInterface` class that users can use to set parameters for different Gern-annotated functions
- a `gen` method that takes in
  - `M`, a Pytorch NN module to generate an optimized version of
  - `torch_to_gern`, a map of functions to `FnInterface` instances (which keeps track of the Gern annotation of each function and additional arguments)
  - `tile_rows`, a parameter (for testing) that sets the row tiling of the generated loops. This should be removed eventually and changed to some calculation based on the size of the tensors.

When `gen` is called, it creates an optimized version of the module `M` passed in and returns it.

To do this, it traces through all of the nodes in the call graph. If it finds a contiguous section of method calls that are defined in `torch_to_gern` (which means they can be replaced by calls to Fern-annotated functions), it will

- create a gern Composable with this specific set of calls, passing in the correct arguments
- create a Gern Runner to run this specific set of calls
- compile the Composable with the Runner
- remove these nodes out of the PyTorch traced graph and replace it with a single call to an external function -- `gern_function_call`

When the `gern_function_call` node is called, it will receive the Runner (which has compiled the program already), the input and output `AbtractDataTypePtr`s, any additional arguments to be passed to Runner::evaluate, and the actual inputs. Then, it can call `Runner::evaluate`, passing in the correct arguments, and return the output.

In this way, the custom `torch.compile` backend traces through the entire forward pass of the NN module and replaces sets of call with an equivalent call into Gern code.

## Examples

- `python/python_test.py` contains an example of using the Python bindings to write a Gern program (in this case, it adds 2 to every element in the matrix)

- `python/pytorch_ex.py` contains various tests and benchmarks for using the `generate_torch_compile_pipeline.py` utilities to optimize a PyTorch NN Module.

## Benchmarks

I also created some benchmarks with the goal of showing that fused code with Gern is faster than the unfused version.

The `benchmark` directory uses Google Benchmark to run these benchmarks.

Currently implemented:

- In `main.cpp`, there are benchmarks for the fusion of 2 matrix multiplications (A @ B @ C), for a variety of different matrix sizes and tile sizes.
  - Run with `./build/dev/benchmark/gern_benchmarks --benchmark_out=benchmark/results.json --benchmark_out_format=json` to write the results to `results.json`
  - Then the results can be visualized by running `python3 plot.py`.

## Todos / Improvements

- Right now, only arguments to functions are properly handled and kwargs are not. There also needs to be flexibility between the order arguments are passed in a pytorch function and the order arguments into Gern (because some Gern functions might expect a slightly different argument order).
- Adding more annotations to the `MatrixCPU` interface. Right now, the `MatrixCPU` interface uses a custom implementation of a 2D Matrix (found in `test/library/matrix/impl/cpu-matrix.h`). An idea to make it easier to implement different operations is to directly use libtorch (C++ library for Pytorch) tensors and operations on these tensors. An example of doing this is in the `MatrixCPU4Dim` interface (also in `test/library/matrix/impl/cpu-matrix.h`).
- For benchmarks, the currently implemented benchmarks will only benchmark the C++ version of Gern
  - More benchmarks need to be added to benchmark Gern from the python side (using python bindings and also the `generate_torch_compile_pipeline.py` torch.compile backend)
  - However, ran into issues where the Pytorch implementation of some Pytorch functions are faster than the libtorch (C++) implementation. For example, benchmarks shpwed that `torch.add` (pytorch) was much faster than `torch::add_out` (libtorch). So this needs to be considered when benchmarking the torch.compiled code against raw Python code.
