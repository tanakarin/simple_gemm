#pragma once

#include <cstddef>
#include <vector>
#include <cassert>
#include <type_traits>

namespace sg {

// Row-major 固定のテンプレートGEMM（混在型対応）
template <typename TA, typename TB, typename TC,
          typename Acc = std::common_type_t<TA, TB, TC>>
inline void gemm(const TA* A, const TB* B, TC* C,
                 std::size_t M, std::size_t N, std::size_t K,
                 Acc alpha = Acc(1), Acc beta = Acc(0)) {
  assert(A && B && C);

  if (beta != Acc(1)) {
    for (std::size_t i = 0; i < M; ++i) {
      for (std::size_t j = 0; j < N; ++j) {
        const Acc c_ij = static_cast<Acc>(C[i * N + j]);
        C[i * N + j] = static_cast<TC>(beta * c_ij);
      }
    }
  }

  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t p = 0; p < K; ++p) {
      const Acc a_ip = alpha * static_cast<Acc>(A[i * K + p]);
      for (std::size_t j = 0; j < N; ++j) {
        const Acc bij = static_cast<Acc>(B[p * N + j]);
        const Acc cij = static_cast<Acc>(C[i * N + j]);
        C[i * N + j] = static_cast<TC>(cij + a_ip * bij);
      }
    }
  }
}

template <typename TA, typename TB, typename TC,
          typename Acc = std::common_type_t<TA, TB, TC>>
inline void gemm(const std::vector<TA>& A,
                 const std::vector<TB>& B,
                 std::vector<TC>& C,
                 std::size_t M, std::size_t N, std::size_t K,
                 Acc alpha = Acc(1), Acc beta = Acc(0)) {
  C.resize(M * N);
  gemm<TA, TB, TC, Acc>(A.data(), B.data(), C.data(), M, N, K, alpha, beta);
}

} // namespace sg
 
