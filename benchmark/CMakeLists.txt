cmake_minimum_required(VERSION 3.30)
project(gern_benchmarks)

find_package(benchmark REQUIRED)

add_executable(gern_benchmarks main.cpp)
include_directories(${PROJECT_BINARY_DIR}/test
"/Users/maggie/.pyenv/versions/3.12.5/lib/python3.12/site-packages/torch/include"
"/Users/maggie/.pyenv/versions/3.12.5/lib/python3.12/site-packages/torch/include/torch/csrc/api/include"
)

target_link_directories(gern_benchmarks PRIVATE
"/Users/maggie/.pyenv/versions/3.12.5/lib/python3.12/site-packages/torch/lib")
target_link_libraries(gern_benchmarks PRIVATE 
Gern::Gern
benchmark::benchmark
torch
torch_cpu
c10
)
