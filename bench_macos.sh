#!/bin/sh

set -eux

export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_DYNAMIC=false

export OMP_NUM_THREADS=12

bin/GMPBenchmarkClang --benchmark_out=GMP-Clang-12.json
bin/MPFRBenchmarkClang --benchmark_out=MPFR-Clang-12.json
bin/CLNBenchmarkClang --benchmark_out=CLN-Clang-12.json
bin/FLINTBenchmarkClang --benchmark_out=FLINT-Clang-12.json
bin/BoostBenchmarkClang --benchmark_out=Boost-Clang-12.json
bin/QDBenchmarkClang --benchmark_out=QD-Clang-12.json
bin/CamparyBenchmarkClang --benchmark_out=Campary-Clang-12.json
bin/MultiFloatsARMBenchmarkClang --benchmark_out=MultiFloatsARM-Clang-12.json

export OMP_NUM_THREADS=6

bin/GMPBenchmarkClang --benchmark_out=GMP-Clang-6.json
bin/MPFRBenchmarkClang --benchmark_out=MPFR-Clang-6.json
bin/CLNBenchmarkClang --benchmark_out=CLN-Clang-6.json
bin/FLINTBenchmarkClang --benchmark_out=FLINT-Clang-6.json
bin/BoostBenchmarkClang --benchmark_out=Boost-Clang-6.json
bin/QDBenchmarkClang --benchmark_out=QD-Clang-6.json
bin/CamparyBenchmarkClang --benchmark_out=Campary-Clang-6.json
bin/MultiFloatsARMBenchmarkClang --benchmark_out=MultiFloatsARM-Clang-6.json
