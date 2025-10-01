#include <iostream>
#include <vector>
#include "gemm.hpp"

int main() {
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
  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      std::cout << C[i*N + j] << (j+1==N?"\n":" ");
    }
  }
  return 0;
}
