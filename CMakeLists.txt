cmake_minimum_required(VERSION 3.15)
project(NeuralNetworkEngine VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(third_party/BitMth)

add_library(nn_core INTERFACE)

target_include_directories(nn_core INTERFACE include)

target_link_libraries(nn_core INTERFACE BitMth)

add_executable(nn_app examples/main.cpp)

target_link_libraries(nn_app PRIVATE nn_core)

