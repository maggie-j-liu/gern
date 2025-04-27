#include <benchmark/benchmark.h>
#include "compose/composable.h"
#include "compose/runner.h"
#include "config.h"
#include "library/matrix/annot/cpu-matrix.h"
#include "library/matrix/impl/cpu-matrix.h"
#include "test-utils.h"
using namespace gern;

int64_t w = 1;
int64_t x = 1;
int64_t y = 1024;
int64_t z = 512;

static void BM_AddTiled(benchmark::State& state) {
    auto inputDS = AbstractDataTypePtr(new const annot::MatrixCPU4Dim("input_con"));
    auto intermediate = AbstractDataTypePtr(new const annot::MatrixCPU4Dim("intermediate"));
    auto outputDS = AbstractDataTypePtr(new const annot::MatrixCPU4Dim("output_con"));

    annot::MatrixAddCPU4D add;
    Variable l_w("l_w");
    Variable l_x("l_x");
    Variable l_y("l_y");
    Variable l_z("l_z");

    Composable program = {
        (Tile(outputDS["dims[0]"], l_w))(
            (Tile(outputDS["dims[1]"], l_x))(
                Tile(outputDS["dims[2]"], l_y)(
                    Tile(outputDS["dims[3]"], l_z)(
                        add(inputDS, intermediate),
                        add(intermediate, outputDS)
                    )
                )
            )
        )};

    Runner run(program);

    run.compile(test::cpuRunner(std::vector<std::string>{"matrix"}));

    

    impl::MatrixCPU4Dim a(w, x, y, z);
    a.vvals(2.0f);
    impl::MatrixCPU4Dim b(w, x, y, z);

    int64_t l_w_val = w;
    int64_t l_x_val = x;
    int64_t l_y_val = 512;
    int64_t l_z_val = z;

    for (auto _ : state) {
        run.evaluate({
            {inputDS.getName(), &a},
            {outputDS.getName(), &b},
            {l_w.getName(), &l_w_val},
            {l_x.getName(), &l_x_val},
            {l_y.getName(), &l_y_val},
            {l_z.getName(), &l_z_val},
        });
    }

    a.destroy();
    b.destroy();
}

static void BM_AddUnTiled(benchmark::State& state) {
    auto inputDS = AbstractDataTypePtr(new const annot::MatrixCPU4Dim("input_con"));
    auto intermediate = AbstractDataTypePtr(new const annot::MatrixCPU4Dim("intermediate"));
    auto outputDS = AbstractDataTypePtr(new const annot::MatrixCPU4Dim("output_con"));

    annot::MatrixAddCPU4D add;
    Variable l_w("l_w");
    Variable l_x("l_x");
    Variable l_y("l_y");
    Variable l_z("l_z");

    Composable program = {
        (Tile(outputDS["dims[0]"], l_w))(
            (Tile(outputDS["dims[1]"], l_x))(
                Tile(outputDS["dims[2]"], l_y)(
                    Tile(outputDS["dims[3]"], l_z)(
                        add(inputDS, intermediate),
                        add(intermediate, outputDS)
                    )
                )
            )
        )};

    Runner run(program);

    run.compile(test::cpuRunner(std::vector<std::string>{"matrix"}));

    impl::MatrixCPU4Dim a(w, x, y, z);
    a.vvals(2.0f);
    impl::MatrixCPU4Dim b(w, x, y, z);

    int64_t l_w_val = w;
    int64_t l_x_val = x;
    int64_t l_y_val = y;
    int64_t l_z_val = z;

    for (auto _ : state) {
        run.evaluate({
            {inputDS.getName(), &a},
            {outputDS.getName(), &b},
            {l_w.getName(), &l_w_val},
            {l_x.getName(), &l_x_val},
            {l_y.getName(), &l_y_val},
            {l_z.getName(), &l_z_val},
        });
    }

    a.destroy();
    b.destroy();
}

static void BM_AddRaw(benchmark::State& state) {
    impl::MatrixCPU4Dim a(w, x, y, z);
    a.vvals(2.0f);
    impl::MatrixCPU4Dim inter(w, x, y, z);
    impl::MatrixCPU4Dim b(w, x, y, z);

    for (auto _ : state) {
        add(a, inter);
        add(inter, b);
    }

    a.destroy();
    inter.destroy();
    b.destroy();
}

BENCHMARK(BM_AddTiled);
BENCHMARK(BM_AddUnTiled);
BENCHMARK(BM_AddRaw);

BENCHMARK_MAIN();

