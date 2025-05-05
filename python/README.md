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
  - `M`, a Pytorch module to generate an optimized version of
  - `torch_to_gern`, a map of functions to `FnInterface` instances (which keeps track of the Gern annotation of each function and additional arguments)
  - `tile_rows`, a parameter (for testing) that sets the row tiling of the generated loops. This should be removed eventually and changed to some calculation based on the size of the tensors.
