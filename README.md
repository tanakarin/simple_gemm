# simple_gemm

Row-major 専用のシンプルな GEMM 実装 (C++17)。

- C = alpha * A(MxK) * B(KxN) + beta * C(MxN)
- 行列は Row-major 固定
- 純粋な三重ループ実装（最適化は未実装）

## 環境
- CMake
- g++
- GoogleTest（ソース一式）。本プロジェクトでは `test/CMakeLists.txt` から `add_subdirectory` で取り込みます。

## ディレクトリ構成
- `include/` ヘッダ (`gemm.hpp`)
- `src/` 実装 (`gemm.cpp`) とサンプル (`main.cpp`)
- `test/` 単体テスト（GoogleTest）
- `CMakeLists.txt` ルートCMake

## quick start（相対パスのみ・どこからでも実行可能）
```bash
# どのディレクトリからでも:
cmake -S . -B build
cmake --build build -j
```

## test（相対パスのみ）
```bash
# build ディレクトリでテストを実行
ctest --test-dir build --output-on-failure
```

## GoogleTest の場所（GTEST_ROOT）の指定（絶対パス不要）
以下の優先順で自動的に探索します:
1. 環境変数 `GTEST_ROOT`（または `-DGTEST_ROOT=...` 指定）
2. `find_package(GTest)`（システムに導入済みの場合）

```bash
# 例) 環境変数を使う（相対でも可）
export GTEST_ROOT=third_party/googletest
cmake -S . -B build -DGTEST_ROOT=$GTEST_ROOT
cmake --build build -j
ctest --test-dir build --output-on-failure
```

ヒント: `GTEST_ROOT` は `test/CMakeLists.txt` で扱っています。指定ディレクトリ直下に googletest の `CMakeLists.txt` が必要です。

## 実行ファイルの動かし方
```bash
./build/simple_gemm_example
```
出力例:
```
58 64
139 154
```
