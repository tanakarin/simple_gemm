cmake_minimum_required(VERSION 3.16)
project(simple_gemm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_WARNINGS "Enable warnings" ON)
if(ENABLE_WARNINGS)
  if(MSVC)
    add_compile_options(/W4 /permissive-)
  else()
    add_compile_options(-Wall -Wextra -Wpedantic -Werror)
  endif()
endif()

add_library(simple_gemm STATIC)
target_include_directories(simple_gemm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_sources(simple_gemm PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src/gemm.cpp
)

add_executable(simple_gemm_example ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)
target_link_libraries(simple_gemm_example PRIVATE simple_gemm)

enable_testing()
add_subdirectory(test)


