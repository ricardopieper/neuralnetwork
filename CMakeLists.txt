cmake_minimum_required(VERSION 3.3)
project(blastest)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -Wall -DWITH_OPENBLAS -lopenblas")

include_directories("/home/ricardo/blastest/libs/FLENS")

set(SOURCE_FILES
    src/DigitClassifier.cpp
    src/DigitClassifier.hpp
    src/FlensOctave.cpp
    src/FlensOctave.hpp
    src/NNDefs.h
    src/fmincg.h
    src/fmincg.cpp
    src/Test.cpp)

add_executable(blastest ${SOURCE_FILES})