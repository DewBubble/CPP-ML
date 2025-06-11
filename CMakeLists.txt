cmake_minimum_required(VERSION 4.0.0)
set(CMAKE_BUILD_TYPE Debug)
project(21-5 VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_C_COMPILER "C:/Program Files/MSYS2/mingw64/bin/gcc.exe")
set(CMAKE_CXX_COMPILER "C:/Program Files/MSYS2/mingw64/bin/g++.exe")

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB SRC_FILES "src/*.cpp")
add_executable(ETL ${SRC_FILES})
#add_library(ETL SHARED ${SRC_FILES})
target_include_directories(ETL PUBLIC "include")
target_link_libraries(ETL PRIVATE -lstdc++exp )