#!/bin/sh

set -eux

export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_DYNAMIC=false

export OMP_NUM_THREADS=32

bin/GMPBenchmarkGCC --benchmark_out=GMP-GCC-32.json
bin/GMPBenchmarkClang --benchmark_out=GMP-Clang-32.json
bin/MPFRBenchmarkGCC --benchmark_out=MPFR-GCC-32.json
bin/MPFRBenchmarkClang --benchmark_out=MPFR-Clang-32.json
bin/CLNBenchmarkGCC --benchmark_out=CLN-GCC-32.json
bin/CLNBenchmarkClang --benchmark_out=CLN-Clang-32.json
bin/FLINTBenchmarkGCC --benchmark_out=FLINT-GCC-32.json
bin/FLINTBenchmarkClang --benchmark_out=FLINT-Clang-32.json
bin/QuadMathBenchmarkGCC --benchmark_out=QuadMath-GCC-32.json
bin/QuadMathBenchmarkClang --benchmark_out=QuadMath-Clang-32.json
bin/BoostBenchmarkGCC --benchmark_out=Boost-GCC-32.json
bin/BoostBenchmarkClang --benchmark_out=Boost-Clang-32.json
bin/QDBenchmarkGCC --benchmark_out=QD-GCC-32.json
bin/QDBenchmarkClang --benchmark_out=QD-Clang-32.json
bin/CamparyBenchmarkGCC --benchmark_out=Campary-GCC-32.json
bin/CamparyBenchmarkClang --benchmark_out=Campary-Clang-32.json
bin/MultiFloatsBenchmarkGCC --benchmark_out=MultiFloats-GCC-32.json
bin/MultiFloatsBenchmarkClang --benchmark_out=MultiFloats-Clang-32.json

export OMP_NUM_THREADS=16

bin/GMPBenchmarkGCC --benchmark_out=GMP-GCC-16.json
bin/GMPBenchmarkClang --benchmark_out=GMP-Clang-16.json
bin/MPFRBenchmarkGCC --benchmark_out=MPFR-GCC-16.json
bin/MPFRBenchmarkClang --benchmark_out=MPFR-Clang-16.json
bin/CLNBenchmarkGCC --benchmark_out=CLN-GCC-16.json
bin/CLNBenchmarkClang --benchmark_out=CLN-Clang-16.json
bin/FLINTBenchmarkGCC --benchmark_out=FLINT-GCC-16.json
bin/FLINTBenchmarkClang --benchmark_out=FLINT-Clang-16.json
bin/QuadMathBenchmarkGCC --benchmark_out=QuadMath-GCC-16.json
bin/QuadMathBenchmarkClang --benchmark_out=QuadMath-Clang-16.json
bin/BoostBenchmarkGCC --benchmark_out=Boost-GCC-16.json
bin/BoostBenchmarkClang --benchmark_out=Boost-Clang-16.json
bin/QDBenchmarkGCC --benchmark_out=QD-GCC-16.json
bin/QDBenchmarkClang --benchmark_out=QD-Clang-16.json
bin/CamparyBenchmarkGCC --benchmark_out=Campary-GCC-16.json
bin/CamparyBenchmarkClang --benchmark_out=Campary-Clang-16.json
bin/MultiFloatsBenchmarkGCC --benchmark_out=MultiFloats-GCC-16.json
bin/MultiFloatsBenchmarkClang --benchmark_out=MultiFloats-Clang-16.json

export OMP_NUM_THREADS=8

bin/GMPBenchmarkGCC --benchmark_out=GMP-GCC-8.json
bin/GMPBenchmarkClang --benchmark_out=GMP-Clang-8.json
bin/MPFRBenchmarkGCC --benchmark_out=MPFR-GCC-8.json
bin/MPFRBenchmarkClang --benchmark_out=MPFR-Clang-8.json
bin/CLNBenchmarkGCC --benchmark_out=CLN-GCC-8.json
bin/CLNBenchmarkClang --benchmark_out=CLN-Clang-8.json
bin/FLINTBenchmarkGCC --benchmark_out=FLINT-GCC-8.json
bin/FLINTBenchmarkClang --benchmark_out=FLINT-Clang-8.json
bin/QuadMathBenchmarkGCC --benchmark_out=QuadMath-GCC-8.json
bin/QuadMathBenchmarkClang --benchmark_out=QuadMath-Clang-8.json
bin/BoostBenchmarkGCC --benchmark_out=Boost-GCC-8.json
bin/BoostBenchmarkClang --benchmark_out=Boost-Clang-8.json
bin/QDBenchmarkGCC --benchmark_out=QD-GCC-8.json
bin/QDBenchmarkClang --benchmark_out=QD-Clang-8.json
bin/CamparyBenchmarkGCC --benchmark_out=Campary-GCC-8.json
bin/CamparyBenchmarkClang --benchmark_out=Campary-Clang-8.json
bin/MultiFloatsBenchmarkGCC --benchmark_out=MultiFloats-GCC-8.json
bin/MultiFloatsBenchmarkClang --benchmark_out=MultiFloats-Clang-8.json

export OMP_NUM_THREADS=4

bin/GMPBenchmarkGCC --benchmark_out=GMP-GCC-4.json
bin/GMPBenchmarkClang --benchmark_out=GMP-Clang-4.json
bin/MPFRBenchmarkGCC --benchmark_out=MPFR-GCC-4.json
bin/MPFRBenchmarkClang --benchmark_out=MPFR-Clang-4.json
bin/CLNBenchmarkGCC --benchmark_out=CLN-GCC-4.json
bin/CLNBenchmarkClang --benchmark_out=CLN-Clang-4.json
bin/FLINTBenchmarkGCC --benchmark_out=FLINT-GCC-4.json
bin/FLINTBenchmarkClang --benchmark_out=FLINT-Clang-4.json
bin/QuadMathBenchmarkGCC --benchmark_out=QuadMath-GCC-4.json
bin/QuadMathBenchmarkClang --benchmark_out=QuadMath-Clang-4.json
bin/BoostBenchmarkGCC --benchmark_out=Boost-GCC-4.json
bin/BoostBenchmarkClang --benchmark_out=Boost-Clang-4.json
bin/QDBenchmarkGCC --benchmark_out=QD-GCC-4.json
bin/QDBenchmarkClang --benchmark_out=QD-Clang-4.json
bin/CamparyBenchmarkGCC --benchmark_out=Campary-GCC-4.json
bin/CamparyBenchmarkClang --benchmark_out=Campary-Clang-4.json
bin/MultiFloatsBenchmarkGCC --benchmark_out=MultiFloats-GCC-4.json
bin/MultiFloatsBenchmarkClang --benchmark_out=MultiFloats-Clang-4.json

export OMP_NUM_THREADS=2

bin/GMPBenchmarkGCC --benchmark_out=GMP-GCC-2.json
bin/GMPBenchmarkClang --benchmark_out=GMP-Clang-2.json
bin/MPFRBenchmarkGCC --benchmark_out=MPFR-GCC-2.json
bin/MPFRBenchmarkClang --benchmark_out=MPFR-Clang-2.json
bin/CLNBenchmarkGCC --benchmark_out=CLN-GCC-2.json
bin/CLNBenchmarkClang --benchmark_out=CLN-Clang-2.json
bin/FLINTBenchmarkGCC --benchmark_out=FLINT-GCC-2.json
bin/FLINTBenchmarkClang --benchmark_out=FLINT-Clang-2.json
bin/QuadMathBenchmarkGCC --benchmark_out=QuadMath-GCC-2.json
bin/QuadMathBenchmarkClang --benchmark_out=QuadMath-Clang-2.json
bin/BoostBenchmarkGCC --benchmark_out=Boost-GCC-2.json
bin/BoostBenchmarkClang --benchmark_out=Boost-Clang-2.json
bin/QDBenchmarkGCC --benchmark_out=QD-GCC-2.json
bin/QDBenchmarkClang --benchmark_out=QD-Clang-2.json
bin/CamparyBenchmarkGCC --benchmark_out=Campary-GCC-2.json
bin/CamparyBenchmarkClang --benchmark_out=Campary-Clang-2.json
bin/MultiFloatsBenchmarkGCC --benchmark_out=MultiFloats-GCC-2.json
bin/MultiFloatsBenchmarkClang --benchmark_out=MultiFloats-Clang-2.json

export OMP_NUM_THREADS=1

bin/GMPBenchmarkGCC --benchmark_out=GMP-GCC-1.json
bin/GMPBenchmarkClang --benchmark_out=GMP-Clang-1.json
bin/MPFRBenchmarkGCC --benchmark_out=MPFR-GCC-1.json
bin/MPFRBenchmarkClang --benchmark_out=MPFR-Clang-1.json
bin/CLNBenchmarkGCC --benchmark_out=CLN-GCC-1.json
bin/CLNBenchmarkClang --benchmark_out=CLN-Clang-1.json
bin/FLINTBenchmarkGCC --benchmark_out=FLINT-GCC-1.json
bin/FLINTBenchmarkClang --benchmark_out=FLINT-Clang-1.json
bin/QuadMathBenchmarkGCC --benchmark_out=QuadMath-GCC-1.json
bin/QuadMathBenchmarkClang --benchmark_out=QuadMath-Clang-1.json
bin/BoostBenchmarkGCC --benchmark_out=Boost-GCC-1.json
bin/BoostBenchmarkClang --benchmark_out=Boost-Clang-1.json
bin/QDBenchmarkGCC --benchmark_out=QD-GCC-1.json
bin/QDBenchmarkClang --benchmark_out=QD-Clang-1.json
bin/CamparyBenchmarkGCC --benchmark_out=Campary-GCC-1.json
bin/CamparyBenchmarkClang --benchmark_out=Campary-Clang-1.json
bin/MultiFloatsBenchmarkGCC --benchmark_out=MultiFloats-GCC-1.json
bin/MultiFloatsBenchmarkClang --benchmark_out=MultiFloats-Clang-1.json
