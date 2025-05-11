#include <benchmark/benchmark.h>
#include "compose/composable.h"
#include "compose/runner.h"
#include "config.h"
#include "library/matrix/annot/cpu-matrix.h"
#include "library/matrix/impl/cpu-matrix.h"
#include "test-utils.h"
using namespace gern;


static void BM_AddTiled(benchmark::State& state) {
    auto inputDS = AbstractDataTypePtr(new const annot::MatrixCPU4Dim("input_con"));
    auto intermediate = AbstractDataTypePtr(new const annot::MatrixCPU4Dim("intermediate"));
    auto outputDS = AbstractDataTypePtr(new const annot::MatrixCPU4Dim("output_con"));

    int64_t w = state.range(0);
    int64_t x = state.range(1);
    int64_t y = state.range(2);
    int64_t z = state.range(3);

    annot::MatrixAddCPU4D add;
    Variable l_w("l_w");
    Variable l_x("l_x");
    Variable l_y("l_y");
    Variable l_z("l_z");
    Variable increment("incr", Datatype::Float32);

    Composable program = {
        (Tile(outputDS["dims[0]"], l_w))(
            (Tile(outputDS["dims[1]"], l_x))(
                Tile(outputDS["dims[2]"], l_y)(
                    Tile(outputDS["dims[3]"], l_z)(
                        add(inputDS, increment, intermediate),
                        add(intermediate, increment, outputDS)
                    )
                )
            )
        )};

    Runner run(program);

    run.compile(test::cpuRunner(std::vector<std::string>{"matrix"}));


    impl::MatrixCPU4Dim a(w, x, y, z);
    a.random_fill();
    impl::MatrixCPU4Dim b(w, x, y, z);

    int64_t l_w_val = w;
    int64_t l_x_val = x;
    int64_t l_y_val = 512;
    int64_t l_z_val = z;
    float incr_val = 1;

    for (auto _ : state) {
        run.evaluate({
            {inputDS.getName(), &a},
            {outputDS.getName(), &b},
            {l_w.getName(), &l_w_val},
            {l_x.getName(), &l_x_val},
            {l_y.getName(), &l_y_val},
            {l_z.getName(), &l_z_val},
            {increment.getName(), &incr_val}
        });
    }

    a.destroy();
    b.destroy();
}

static void BM_AddUnTiled(benchmark::State& state) {
    auto inputDS = AbstractDataTypePtr(new const annot::MatrixCPU4Dim("input_con"));
    auto intermediate = AbstractDataTypePtr(new const annot::MatrixCPU4Dim("intermediate"));
    auto outputDS = AbstractDataTypePtr(new const annot::MatrixCPU4Dim("output_con"));

    int64_t w = state.range(0);
    int64_t x = state.range(1);
    int64_t y = state.range(2);
    int64_t z = state.range(3);

    annot::MatrixAddCPU4D add;
    Variable l_w("l_w");
    Variable l_x("l_x");
    Variable l_y("l_y");
    Variable l_z("l_z");
    Variable increment("incr", Datatype::Float32);

    Composable program = {
        (Tile(outputDS["dims[0]"], l_w))(
            (Tile(outputDS["dims[1]"], l_x))(
                Tile(outputDS["dims[2]"], l_y)(
                    Tile(outputDS["dims[3]"], l_z)(
                        add(inputDS, increment, intermediate),
                        add(intermediate, increment, outputDS)
                    )
                )
            )
        )};

    Runner run(program);

    run.compile(test::cpuRunner(std::vector<std::string>{"matrix"}));

    impl::MatrixCPU4Dim a(w, x, y, z);
    a.random_fill();
    impl::MatrixCPU4Dim b(w, x, y, z);

    int64_t l_w_val = w;
    int64_t l_x_val = x;
    int64_t l_y_val = y;
    int64_t l_z_val = z;
    float incr_val = 1;

    for (auto _ : state) {
        run.evaluate({
            {inputDS.getName(), &a},
            {outputDS.getName(), &b},
            {l_w.getName(), &l_w_val},
            {l_x.getName(), &l_x_val},
            {l_y.getName(), &l_y_val},
            {l_z.getName(), &l_z_val},
            {increment.getName(), &incr_val}
        });
    }

    a.destroy();
    b.destroy();
}

static void BM_AddRaw(benchmark::State& state) {
    int64_t w = state.range(0);
    int64_t x = state.range(1);
    int64_t y = state.range(2);
    int64_t z = state.range(3);

    impl::MatrixCPU4Dim a(w, x, y, z);
    a.random_fill();
    impl::MatrixCPU4Dim inter(w, x, y, z);
    impl::MatrixCPU4Dim b(w, x, y, z);

    for (auto _ : state) {
        add(a, 1, inter);
        add(inter, 1, b);
    }

    a.destroy();
    inter.destroy();
    b.destroy();
}

static void BM_Add(benchmark::State& state) {
    int64_t w = state.range(0);
    int64_t x = state.range(1);
    int64_t y = state.range(2);
    int64_t z = state.range(3);

    at::Tensor a = torch::rand({w, x, y, z}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    at::Tensor b = torch::empty({w, x, y, z}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

    torch::NoGradGuard no_grad;

    for (auto _ : state) {
        at::add_out(b, a, 1);
    }
}

static void BM_Add_In_Place(benchmark::State& state) {
    int64_t w = state.range(0);
    int64_t x = state.range(1);
    int64_t y = state.range(2);
    int64_t z = state.range(3);

    at::Tensor a = torch::rand({w, x, y, z}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

    torch::NoGradGuard no_grad;

    for (auto _ : state) {
        a.add_(1);
    }
}

static void BM_Matmul(benchmark::State& state) {
    int64_t w = state.range(0);
    int64_t x = state.range(1);
    int64_t y = state.range(2);
    int64_t z = state.range(3);

    int64_t w1 = state.range(4);
    int64_t x1 = state.range(5);
    int64_t y1 = state.range(6);
    int64_t z1 = state.range(7);

    at::Tensor a = torch::rand({w, x, y, z}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    at::Tensor b = torch::rand({w1, x1, y1, z1}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    at::Tensor c = torch::empty({w, x, y, z1}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

    torch::NoGradGuard no_grad;

    for (auto _ : state) {
        at::matmul_out(c, a, b);
    }
}

static void BM_DoubleMatmul(benchmark::State& state) {
    int64_t w = state.range(0);
    int64_t x = state.range(1);

    int64_t y = state.range(2);
    int64_t z = state.range(3);

    int64_t y1 = state.range(4);
    int64_t z1 = state.range(5);

    int64_t y2 = state.range(6);
    int64_t z2 = state.range(7);

    assert(z == y1);
    assert(z1 == y2);

    at::Tensor a = torch::rand({w, x, y, z}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    at::Tensor b = torch::rand({w, x, y1, z1}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    // at::Tensor c = torch::empty({w, x, y, z1}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    at::Tensor d = torch::empty({w, x, y2, z2}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    at::Tensor e = torch::empty({w, x, y, z2}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

    torch::NoGradGuard no_grad;

    for (auto _ : state) {
        at::Tensor c = at::matmul(a, b);
        at::matmul_out(e, c, d);
    }
}

static void BM_GernDoubleMatmul(benchmark::State& state) {
    auto A_DS = AbstractDataTypePtr(new const annot::MatrixCPU4Dim("A"));
    auto B_DS = AbstractDataTypePtr(new const annot::MatrixCPU4Dim("B"));
    auto C_DS = AbstractDataTypePtr(new const annot::MatrixCPU4Dim("C"));
    auto D_DS = AbstractDataTypePtr(new const annot::MatrixCPU4Dim("D"));
    auto E_DS = AbstractDataTypePtr(new const annot::MatrixCPU4Dim("E"));

    Variable k1("k1");
    Variable k2("k2");
    Variable l_w("l_w");
    Variable l_x("l_x");
    Variable l_y("l_y");
    Variable l_z("l_z");

    annot::MatrixMultiply4D matrix_multiply1;
	matrix_multiply1[{ { "shared_len", k1 }}];
    annot::MatrixMultiply4D matrix_multiply2;
	matrix_multiply2[{ { "shared_len", k2 }}];

    Composable program({
        Tile(E_DS["dims[0]"], l_w)(
            Tile(E_DS["dims[1]"], l_x)(
                Tile(E_DS["dims[2]"], l_y)(
                    Tile(E_DS["dims[3]"], l_z)(
                        matrix_multiply1(A_DS, B_DS, C_DS),
                        matrix_multiply2(C_DS, D_DS, E_DS)
                    )
                )
            )
        )
    });

    Runner run(program);
    run.compile(test::cpuRunner(std::vector<std::string>{"matrix"}));

    int64_t w = state.range(0);
    int64_t x = state.range(1);

    int64_t y = state.range(2);
    int64_t z = state.range(3);

    int64_t y1 = state.range(4);
    int64_t z1 = state.range(5);

    int64_t y2 = state.range(6);
    int64_t z2 = state.range(7);

    int64_t l_w_val = w;
    int64_t l_x_val = x;
    int64_t l_y_val = state.range(8);
    int64_t l_z_val = state.range(9);

    assert(z == y1);
    assert(z1 == y2);

    impl::MatrixCPU4Dim a(w, x, y, z);
    a.random_fill();

    impl::MatrixCPU4Dim b(w, x, y1, z1);
    b.random_fill();

    impl::MatrixCPU4Dim d(w, x, y2, z2);
    d.random_fill();

    impl::MatrixCPU4Dim e(w, x, y, z2);

    for (auto _ : state) {
        run.evaluate({
            {A_DS.getName(), &a},
            {B_DS.getName(), &b},
            {D_DS.getName(), &d},
            {E_DS.getName(), &e},
            {l_w.getName(), &l_w_val},
            {l_x.getName(), &l_x_val},
            {l_y.getName(), &l_y_val},
            {l_z.getName(), &l_z_val},
            {k1.getName(), &z},
            {k2.getName(), &z1}
        });
    }

    a.destroy();
    b.destroy();
    d.destroy();
    e.destroy();
}

// BENCHMARK(BM_AddTiled)->Repetitions(5)->MinTime(4)->MinWarmUpTime(4)->Args({1, 1, 1024, 512})->Args({1, 12, 1024, 64});
// BENCHMARK(BM_AddUnTiled)->Repetitions(5)->MinTime(4)->MinWarmUpTime(4)->Args({1, 1, 1024, 512})->Args({1, 12, 1024, 64});
// BENCHMARK(BM_AddRaw)->Repetitions(5)->MinTime(4)->MinWarmUpTime(4)->Args({1, 1, 1024, 512})->Args({1, 12, 1024, 64});

// BENCHMARK(BM_Add)->Repetitions(5)->MinTime(4)->MinWarmUpTime(4)->Args({1, 1, 1024, 512});
// BENCHMARK(BM_Add_In_Place)->Repetitions(5)->MinTime(4)->MinWarmUpTime(4)->Args({1, 1, 1024, 512});
// BENCHMARK(BM_Matmul)->Repetitions(5)->MinTime(4)->MinWarmUpTime(4)->Args({1, 1, 1024, 512, 1, 1, 512, 1024});
// BENCHMARK(BM_DoubleMatmul)->Repetitions(5)->MinTime(4)->MinWarmUpTime(4)
    // ->Args({1, 1, 512, 512, 512, 512, 512, 512})
    // ->Args({1, 1, 1024, 512, 512, 1024, 1024, 512})
    // ->Args({1, 1, 1024, 1024, 1024, 1024, 1024, 1024});
BENCHMARK(BM_GernDoubleMatmul)->Repetitions(5)->MinTime(4)->MinWarmUpTime(4)
    ->Args({1, 1, 512, 512, 512, 512, 512, 512, 256, 256})
    ->Args({1, 1, 512, 512, 512, 512, 512, 512, 256, 512})
    ->Args({1, 1, 512, 512, 512, 512, 512, 512, 512, 512})

	->Args({1, 1, 1024, 256, 256, 1024, 1024, 256, 128, 128})
	->Args({1, 1, 1024, 256, 256, 1024, 1024, 256, 128, 256})
	->Args({1, 1, 1024, 256, 256, 1024, 1024, 256, 256, 128})
	->Args({1, 1, 1024, 256, 256, 1024, 1024, 256, 256, 256})
	->Args({1, 1, 1024, 256, 256, 1024, 1024, 256, 512, 128})
	->Args({1, 1, 1024, 256, 256, 1024, 1024, 256, 512, 256})
	->Args({1, 1, 1024, 256, 256, 1024, 1024, 256, 1024, 128})
	->Args({1, 1, 1024, 256, 256, 1024, 1024, 256, 1024, 256})

	->Args({1, 1, 1024, 512, 512, 1024, 1024, 512, 128, 256})
    ->Args({1, 1, 1024, 512, 512, 1024, 1024, 512, 128, 512})
    ->Args({1, 1, 1024, 512, 512, 1024, 1024, 512, 256, 256})
    ->Args({1, 1, 1024, 512, 512, 1024, 1024, 512, 256, 512})
    ->Args({1, 1, 1024, 512, 512, 1024, 1024, 512, 512, 256})
    ->Args({1, 1, 1024, 512, 512, 1024, 1024, 512, 512, 512})
    ->Args({1, 1, 1024, 512, 512, 1024, 1024, 512, 1024, 256})
    ->Args({1, 1, 1024, 512, 512, 1024, 1024, 512, 1024, 512})

    ->Args({1, 1, 1024, 1024, 1024, 1024, 1024, 1024, 256, 256})
    ->Args({1, 1, 1024, 1024, 1024, 1024, 1024, 1024, 512, 512})
    ->Args({1, 1, 1024, 1024, 1024, 1024, 1024, 1024, 512, 1024})
    ->Args({1, 1, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024})

	->Args({1, 1, 2048, 256, 256, 2048, 2048, 256, 256, 128})
	->Args({1, 1, 2048, 256, 256, 2048, 2048, 256, 256, 256})
	->Args({1, 1, 2048, 256, 256, 2048, 2048, 256, 512, 128})
	->Args({1, 1, 2048, 256, 256, 2048, 2048, 256, 512, 256})
	->Args({1, 1, 2048, 256, 256, 2048, 2048, 256, 1024, 128})
	->Args({1, 1, 2048, 256, 256, 2048, 2048, 256, 1024, 256})
	->Args({1, 1, 2048, 256, 256, 2048, 2048, 256, 2048, 128})
	->Args({1, 1, 2048, 256, 256, 2048, 2048, 256, 2048, 256})

	->Args({1, 1, 2048, 512, 512, 2048, 2048, 512, 256, 256})
	->Args({1, 1, 2048, 512, 512, 2048, 2048, 512, 256, 512})
	->Args({1, 1, 2048, 512, 512, 2048, 2048, 512, 512, 256})
	->Args({1, 1, 2048, 512, 512, 2048, 2048, 512, 512, 512})
	->Args({1, 1, 2048, 512, 512, 2048, 2048, 512, 1024, 256})
	->Args({1, 1, 2048, 512, 512, 2048, 2048, 512, 1024, 512})
	->Args({1, 1, 2048, 512, 512, 2048, 2048, 512, 2048, 256})
	->Args({1, 1, 2048, 512, 512, 2048, 2048, 512, 2048, 512})

	->Args({1, 1, 2048, 2048, 2048, 2048, 2048, 2048, 256, 256})
	->Args({1, 1, 2048, 2048, 2048, 2048, 2048, 2048, 512, 512})
	->Args({1, 1, 2048, 2048, 2048, 2048, 2048, 2048, 512, 1024})
	->Args({1, 1, 2048, 2048, 2048, 2048, 2048, 2048, 512, 2048})
	->Args({1, 1, 2048, 2048, 2048, 2048, 2048, 2048, 1024, 512})
	->Args({1, 1, 2048, 2048, 2048, 2048, 2048, 2048, 1024, 1024})
	->Args({1, 1, 2048, 2048, 2048, 2048, 2048, 2048, 1024, 2048})
	->Args({1, 1, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 512})
	->Args({1, 1, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 1024})
	->Args({1, 1, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048});

BENCHMARK_MAIN();

