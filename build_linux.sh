#!/bin/sh

mkdir -p bin

GCC_WARNING_FLAGS="-Wall -Wextra -Wpedantic -pedantic-errors -Wshadow \
    -Wconversion -Warith-conversion -Wdouble-promotion \
    -Wcast-qual -Wuseless-cast -Winit-self -Wlogical-op \
    -Wformat=2 -Wstrict-overflow=5 -Wdisabled-optimization -Winline "

CLANG_WARNING_FLAGS="-Weverything -Wno-c++98-compat-pedantic \
    -Wno-unsafe-buffer-usage -Wno-global-constructors -Wno-alloca"

DIALECT_FLAGS="-std=c++17 -fopenmp \
    -ffp-contract=off -fno-math-errno -fno-exceptions"

OPTIMIZATION_FLAGS="-O3 -march=native -flto \
    -fvisibility=hidden -fmerge-all-constants"

# g++-14 -Q $GCC_WARNING_FLAGS --help=c++,warning | grep -v "\[enabled\]"

g++-14 $DIALECT_FLAGS $GCC_WARNING_FLAGS $OPTIMIZATION_FLAGS \
    -fwhole-program gmp/main.cpp -lgmp -lbenchmark -lbenchmark_main \
    -o bin/GMPBenchmarkGCC

clang++-20 $DIALECT_FLAGS $CLANG_WARNING_FLAGS $OPTIMIZATION_FLAGS \
    gmp/main.cpp -lgmp -lbenchmark -lbenchmark_main \
    -o bin/GMPBenchmarkClang

g++-14 $DIALECT_FLAGS $GCC_WARNING_FLAGS $OPTIMIZATION_FLAGS \
    -fwhole-program mpfr/main.cpp -lmpfr -lbenchmark -lbenchmark_main \
    -o bin/MPFRBenchmarkGCC

clang++-20 $DIALECT_FLAGS $CLANG_WARNING_FLAGS $OPTIMIZATION_FLAGS \
    mpfr/main.cpp -lmpfr -lbenchmark -lbenchmark_main \
    -o bin/MPFRBenchmarkClang
