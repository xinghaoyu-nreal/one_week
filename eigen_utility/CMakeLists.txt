cmake_minimum_required(VERSION 2.8.3)
project(eigen_utility)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++17")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

set(PUBLIC_HEADERS 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    "/usr/include/Eigen"
)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/../lib)

#生成静态库
#add_library(eigen_utility STATIC src/eigen_utility.cpp)
#生成动态库
add_library(eigen_utility SHARED src/eigen_utility.cpp)

target_include_directories(eigen_utility PUBLIC ${PUBLIC_HEADERS})

