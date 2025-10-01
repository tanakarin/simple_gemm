cmake_minimum_required(VERSION 3.16)

add_executable(test_gemm ${CMAKE_CURRENT_SOURCE_DIR}/gemm_test.cpp)

# Try in this order:
# 1) Environment variable GTEST_ROOT (or -DGTEST_ROOT)
# 2) find_package(GTest)

if(NOT GTEST_ROOT AND DEFINED ENV{GTEST_ROOT})
  set(GTEST_ROOT $ENV{GTEST_ROOT})
endif()

if(GTEST_ROOT AND EXISTS ${GTEST_ROOT}/CMakeLists.txt)
  add_subdirectory(${GTEST_ROOT} ${CMAKE_BINARY_DIR}/_deps/googletest EXCLUDE_FROM_ALL)
  set(_use_gtest_targets ON)
else()
  find_package(GTest QUIET)
endif()

if(DEFINED _use_gtest_targets)
  target_link_libraries(test_gemm PRIVATE gtest gtest_main)
elseif(GTest_FOUND)
  target_link_libraries(test_gemm PRIVATE GTest::gtest GTest::gtest_main)
else()
  message(FATAL_ERROR "GoogleTest not found. Install GTest or set GTEST_ROOT to googletest source (directory with CMakeLists.txt).")
endif()

target_include_directories(test_gemm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(test_gemm PRIVATE simple_gemm)

add_test(NAME test_gemm COMMAND test_gemm)