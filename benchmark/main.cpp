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

BENCHMARK(BM_AddTiled)->Repetitions(5)->MinTime(4)->MinWarmUpTime(4)->Args({1, 1, 1024, 512})->Args({1, 12, 1024, 64});
BENCHMARK(BM_AddUnTiled)->Repetitions(5)->MinTime(4)->MinWarmUpTime(4)->Args({1, 1, 1024, 512})->Args({1, 12, 1024, 64});
BENCHMARK(BM_AddRaw)->Repetitions(5)->MinTime(4)->MinWarmUpTime(4)->Args({1, 1, 1024, 512})->Args({1, 12, 1024, 64});

BENCHMARK_MAIN();

