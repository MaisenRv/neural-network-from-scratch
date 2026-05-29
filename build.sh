#!/bin/bash
g++ -std=c++20 -g -O3 -ffast-math -I"$1/include" $(find "$1/src" -type f -name '*.cpp') "$1/main.cpp" -o "$1/bin/$2"
