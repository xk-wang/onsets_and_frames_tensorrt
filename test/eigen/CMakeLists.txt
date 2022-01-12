cmake_minimum_required(VERSION 3.1)
project(eigen)

# Enable C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -mavx -mfma")
add_executable(eigen  eigen.cpp)
target_include_directories(eigen PRIVATE ../../cpp/eigen)