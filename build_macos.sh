#!/bin/sh

set -eux

mkdir -p bin

DIALECT_FLAGS="-std=c++17 -fopenmp -ffp-contract=off -fno-math-errno"

CLANG_WARNING_FLAGS="-Weverything -Wno-float-equal -Wno-c++98-compat-pedantic \
    -Wno-unsafe-buffer-usage -Wno-global-constructors -Wno-alloca"

OPTIMIZATION_FLAGS="-O3 -march=native -flto \
    -fvisibility=hidden -fmerge-all-constants"

# Add Homebrew library paths for macOS
HOMEBREW_FLAGS="-isystem /opt/homebrew/include -L/opt/homebrew/lib"

/opt/homebrew/opt/llvm/bin/clang++ $DIALECT_FLAGS $CLANG_WARNING_FLAGS $OPTIMIZATION_FLAGS $HOMEBREW_FLAGS \
    -isystem./include -I./common \
    gmp/main.cpp -lgmp -lbenchmark -lbenchmark_main \
    -o bin/GMPBenchmarkClang

/opt/homebrew/opt/llvm/bin/clang++ $DIALECT_FLAGS $CLANG_WARNING_FLAGS $OPTIMIZATION_FLAGS $HOMEBREW_FLAGS \
    -isystem./include -I./common \
    mpfr/main.cpp -lmpfr -lbenchmark -lbenchmark_main \
    -o bin/MPFRBenchmarkClang

/opt/homebrew/opt/llvm/bin/clang++ $DIALECT_FLAGS $CLANG_WARNING_FLAGS $OPTIMIZATION_FLAGS $HOMEBREW_FLAGS \
    -isystem./include -I./common \
    cln/main.cpp -lcln -lbenchmark -lbenchmark_main \
    -o bin/CLNBenchmarkClang

/opt/homebrew/opt/llvm/bin/clang++ $DIALECT_FLAGS $CLANG_WARNING_FLAGS $OPTIMIZATION_FLAGS $HOMEBREW_FLAGS \
    -isystem./include -I./common \
    flint/main.cpp -lflint -lbenchmark -lbenchmark_main \
    -o bin/FLINTBenchmarkClang

/opt/homebrew/opt/llvm/bin/clang++ $DIALECT_FLAGS $CLANG_WARNING_FLAGS $OPTIMIZATION_FLAGS $HOMEBREW_FLAGS \
    -isystem./include -I./common \
    boost/main.cpp -lbenchmark -lbenchmark_main \
    -o bin/BoostBenchmarkClang

/opt/homebrew/opt/llvm/bin/clang++ $DIALECT_FLAGS $CLANG_WARNING_FLAGS $OPTIMIZATION_FLAGS $HOMEBREW_FLAGS \
    -isystem./include -I./common \
    qd/main.cpp -lbenchmark -lbenchmark_main \
    -o bin/QDBenchmarkClang

/opt/homebrew/opt/llvm/bin/clang++ $DIALECT_FLAGS $CLANG_WARNING_FLAGS $OPTIMIZATION_FLAGS $HOMEBREW_FLAGS \
    -isystem./include -I./common \
    campary/main.cpp -lbenchmark -lbenchmark_main \
    -o bin/CamparyBenchmarkClang

/opt/homebrew/opt/llvm/bin/clang++ $DIALECT_FLAGS $CLANG_WARNING_FLAGS $OPTIMIZATION_FLAGS $HOMEBREW_FLAGS \
    -isystem./include -I./common \
    multifloats-arm/main.cpp -lbenchmark -lbenchmark_main \
    -o bin/MultiFloatsARMBenchmarkClang
