#!/bin/sh

set -eux

mkdir -p bin

DIALECT_FLAGS="-std=c++17 -fopenmp -ffp-contract=off -fno-math-errno"

GCC_WARNING_FLAGS="-Wall -Wextra -Wpedantic -pedantic-errors -Wshadow \
    -Wconversion -Warith-conversion -Wdouble-promotion \
    -Wcast-qual -Wuseless-cast -Winit-self -Wlogical-op \
    -Wformat=2 -Wstrict-overflow=2 -Wdisabled-optimization -Winline "

CLANG_WARNING_FLAGS="-Weverything -Wno-float-equal -Wno-c++98-compat-pedantic \
    -Wno-unsafe-buffer-usage -Wno-global-constructors -Wno-alloca"

OPTIMIZATION_FLAGS="-O3 -march=native -flto \
    -fvisibility=hidden -fmerge-all-constants"

# g++-14 -Q $GCC_WARNING_FLAGS --help=c++,warning | grep -v "\[enabled\]"

g++-14 $DIALECT_FLAGS $GCC_WARNING_FLAGS $OPTIMIZATION_FLAGS \
    -isystem./include -I./common \
    -fwhole-program gmp/main.cpp -lgmp -lbenchmark -lbenchmark_main \
    -o bin/GMPBenchmarkGCC

clang++-20 $DIALECT_FLAGS $CLANG_WARNING_FLAGS $OPTIMIZATION_FLAGS \
    -isystem./include -I./common \
    gmp/main.cpp -lgmp -lbenchmark -lbenchmark_main \
    -o bin/GMPBenchmarkClang

g++-14 $DIALECT_FLAGS $GCC_WARNING_FLAGS $OPTIMIZATION_FLAGS \
    -isystem./include -I./common \
    -fwhole-program mpfr/main.cpp -lmpfr -lbenchmark -lbenchmark_main \
    -o bin/MPFRBenchmarkGCC

clang++-20 $DIALECT_FLAGS $CLANG_WARNING_FLAGS $OPTIMIZATION_FLAGS \
    -isystem./include -I./common \
    mpfr/main.cpp -lmpfr -lbenchmark -lbenchmark_main \
    -o bin/MPFRBenchmarkClang

g++-14 $DIALECT_FLAGS $GCC_WARNING_FLAGS $OPTIMIZATION_FLAGS \
    -isystem./include -I./common \
    -fwhole-program cln/main.cpp -lcln -lbenchmark -lbenchmark_main \
    -o bin/CLNBenchmarkGCC

clang++-20 $DIALECT_FLAGS $CLANG_WARNING_FLAGS $OPTIMIZATION_FLAGS \
    -isystem./include -I./common \
    cln/main.cpp -lcln -lbenchmark -lbenchmark_main \
    -o bin/CLNBenchmarkClang

g++-14 $DIALECT_FLAGS $GCC_WARNING_FLAGS $OPTIMIZATION_FLAGS \
    -isystem./include -I./common \
    -fwhole-program flint/main.cpp -lflint -lbenchmark -lbenchmark_main \
    -o bin/FLINTBenchmarkGCC

clang++-20 $DIALECT_FLAGS $CLANG_WARNING_FLAGS $OPTIMIZATION_FLAGS \
    -isystem./include -I./common \
    flint/main.cpp -lflint -lbenchmark -lbenchmark_main \
    -o bin/FLINTBenchmarkClang

g++-14 $DIALECT_FLAGS $GCC_WARNING_FLAGS $OPTIMIZATION_FLAGS \
    -isystem./include -I./common \
    -fwhole-program quadmath/main.cpp -lbenchmark -lbenchmark_main \
    -o bin/QuadMathBenchmarkGCC

clang++-20 $DIALECT_FLAGS $CLANG_WARNING_FLAGS $OPTIMIZATION_FLAGS \
    -isystem./include -I./common \
    quadmath/main.cpp -lbenchmark -lbenchmark_main \
    -o bin/QuadMathBenchmarkClang

g++-14 $DIALECT_FLAGS $GCC_WARNING_FLAGS $OPTIMIZATION_FLAGS \
    -isystem./include -I./common \
    -fwhole-program boost/main.cpp -lbenchmark -lbenchmark_main \
    -o bin/BoostBenchmarkGCC

clang++-20 $DIALECT_FLAGS $CLANG_WARNING_FLAGS $OPTIMIZATION_FLAGS \
    -isystem./include -I./common \
    boost/main.cpp -lbenchmark -lbenchmark_main \
    -o bin/BoostBenchmarkClang

g++-14 $DIALECT_FLAGS $GCC_WARNING_FLAGS $OPTIMIZATION_FLAGS \
    -isystem./include -I./common \
    -fwhole-program qd/main.cpp -lbenchmark -lbenchmark_main \
    -o bin/QDBenchmarkGCC

clang++-20 $DIALECT_FLAGS $CLANG_WARNING_FLAGS $OPTIMIZATION_FLAGS \
    -isystem./include -I./common \
    qd/main.cpp -lbenchmark -lbenchmark_main \
    -o bin/QDBenchmarkClang

g++-14 $DIALECT_FLAGS $GCC_WARNING_FLAGS $OPTIMIZATION_FLAGS \
    -isystem./include -I./common \
    -fwhole-program campary/main.cpp -lbenchmark -lbenchmark_main \
    -o bin/CamparyBenchmarkGCC

clang++-20 $DIALECT_FLAGS $CLANG_WARNING_FLAGS $OPTIMIZATION_FLAGS \
    -isystem./include -I./common \
    campary/main.cpp -lbenchmark -lbenchmark_main \
    -o bin/CamparyBenchmarkClang

g++-14 $DIALECT_FLAGS $GCC_WARNING_FLAGS $OPTIMIZATION_FLAGS \
    -isystem./include -I./common \
    -fwhole-program multifloats/main.cpp -lbenchmark -lbenchmark_main \
    -o bin/MultiFloatsBenchmarkGCC

clang++-20 $DIALECT_FLAGS $CLANG_WARNING_FLAGS $OPTIMIZATION_FLAGS \
    -isystem./include -I./common \
    multifloats/main.cpp -lbenchmark -lbenchmark_main \
    -o bin/MultiFloatsBenchmarkClang
