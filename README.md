# simple_gemm
Row-majorのシンプルなGEMM

- C = alpha * A(MxK) * B(KxN) + beta * C(MxN)
- 行列は Row-major

## requirement
- CMake
- g++
- GoogleTest

## quick start
GoogleTest の場所(GTEST_ROOT)の指定が必要
1. 環境変数 `GTEST_ROOT`（または `-DGTEST_ROOT=...` 指定）
2. `find_package(GTest)`（システムに導入済みの場合）

```bash
cmake -S . -B build
cmake --build build -j
```

## test
```bash
ctest --test-dir build --output-on-failure
```