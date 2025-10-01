#include <gtest/gtest.h>
#include "gemm.hpp"
#include <vector>
#include <random>

TEST(GemmTest, Basic2x3x2RowMajor) {
  const std::size_t M = 2, K = 3, N = 2;
  std::vector<float> A = {
    1, 2, 3,
    4, 5, 6
  };
  std::vector<float> B = {
    7, 8,
    9,10,
   11,12
  };
  std::vector<float> C;
  sg::gemm(A, B, C, M, N, K);
  std::vector<float> expected = {
    58, 64,
   139,154
  };
  ASSERT_EQ(C.size(), expected.size());
  for (size_t i = 0; i < C.size(); ++i) {
    EXPECT_FLOAT_EQ(C[i], expected[i]);
  }
}

TEST(GemmTest, WithAlphaBeta) {
  const std::size_t M = 1, K = 2, N = 1;
  std::vector<float> A = {1, 2};
  std::vector<float> B = {3, 4};
  std::vector<float> C = {5};
  sg::gemm(A, B, C, M, N, K, 2.0f, 0.5f);
  EXPECT_FLOAT_EQ(C[0], 24.5f);
}

TEST(GemmTest, MixedTypes_IntFloatToDouble) {
  const std::size_t M = 2, K = 2, N = 2;
  std::vector<int> A = {
    1, 2,
    3, 4
  };
  std::vector<float> B = {
    0.5f, 1.5f,
    -2.0f, 3.0f
  };
  std::vector<double> C(M * N, 1.0);

  // C = 2 * A*B + 0.5 * C
  sg::gemm<int, float, double>(A, B, C, M, N, K, 2.0, 0.5);

  // 手計算
  // A*B = [[1*0.5 + 2*(-2), 1*1.5 + 2*3], [3*0.5 + 4*(-2), 3*1.5 + 4*3]]
  //     = [[0.5 - 4, 1.5 + 6], [1.5 - 8, 4.5 + 12]]
  //     = [[-3.5, 7.5], [-6.5, 16.5]]
  // C' = 2*AB + 0.5*C0 (C0は全1)
  //    = [[-7 + 0.5, 15 + 0.5], [-13 + 0.5, 33 + 0.5]]
  //    = [[-6.5, 15.5], [-12.5, 33.5]]
  std::vector<double> expected = {
    -6.5, 15.5,
    -12.5, 33.5
  };

  for (size_t i = 0; i < C.size(); ++i) {
    EXPECT_NEAR(C[i], expected[i], 1e-12);
  }
}

// 乱数で複数回検証: float 固定型
TEST(GemmTest, RandomFloatMultipleRuns) {
  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

  for (int trial = 0; trial < 10; ++trial) {
    const std::size_t M = 5, K = 7, N = 3;
    std::vector<float> A(M * K), B(K * N), C(M * N, 0.0f), C_ref(M * N, 0.0f);
    for (auto& x : A) x = dist(rng);
    for (auto& x : B) x = dist(rng);

    // 参照実装（ナイーブ）
    for (std::size_t i = 0; i < M; ++i)
      for (std::size_t p = 0; p < K; ++p)
        for (std::size_t j = 0; j < N; ++j)
          C_ref[i*N + j] += A[i*K + p] * B[p*N + j];

    sg::gemm(A, B, C, M, N, K);

    for (std::size_t i = 0; i < M*N; ++i)
      EXPECT_NEAR(C[i], C_ref[i], 1e-4f);
  }
}

// 乱数で複数回検証: 混在型（int×double→float）
TEST(GemmTest, RandomMixed_IntDoubleToFloat) {
  std::mt19937 rng(67890);
  std::uniform_int_distribution<int> dist_i(-3, 3);
  std::uniform_real_distribution<double> dist_d(-1.0, 1.0);

  for (int trial = 0; trial < 10; ++trial) {
    const std::size_t M = 4, K = 6, N = 5;
    std::vector<int> A(M * K);
    std::vector<double> B(K * N);
    std::vector<float> C(M * N, 0.0f), C_ref(M * N, 0.0f);
    for (auto& x : A) x = dist_i(rng);
    for (auto& x : B) x = dist_d(rng);

    // 参照実装（Accはdouble）
    for (std::size_t i = 0; i < M; ++i)
      for (std::size_t p = 0; p < K; ++p)
        for (std::size_t j = 0; j < N; ++j)
          C_ref[i*N + j] = static_cast<float>(static_cast<double>(C_ref[i*N + j]) + static_cast<double>(A[i*K + p]) * B[p*N + j]);

    sg::gemm<int, double, float>(A, B, C, M, N, K);

    for (std::size_t i = 0; i < M*N; ++i)
      EXPECT_NEAR(C[i], C_ref[i], 1e-3f);
  }
}


