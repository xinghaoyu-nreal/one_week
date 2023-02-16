cmake_minimum_required(VERSION 2.8.3)
project(nreal_essential)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++17")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

#ceres
set(Ceres_DIR "/home/nreal/software/ceres2/lib/cmake/Ceres/")
find_package(Ceres)

#OpenCV
set(OpenCV_DIR "/home/nreal/software/opencv-3.4.0/build")
find_package(OpenCV REQUIRED)

add_subdirectory(eigen_utility)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/eigen_utility/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/factor
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
)

add_executable(main 
    src/main.cpp
    src/opencv_utility.cpp
    src/factor/projection.cpp
    src/factor/pose_local_parameterization.cpp
    src/factor/utility.cpp
)
target_link_libraries(main eigen_utility ${OpenCV_LIBS} ${CERES_LIBRARIES})