cmake_minimum_required(VERSION 3.0)

project(ROIALIGN LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 14)

find_package(Torch REQUIRED)
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-D__CUDA_NO_HALF_OPERATORS__")
#MESSAGE( STATUS "this var key = ${CUDA_NVCC_FLAGS}.")

file(GLOB_RECURSE CURRENT_HEADERS  *.h *.hpp *.cuh)
file(GLOB CURRENT_SOURCES  *.cpp *.cu)

source_group("Include" FILES ${CURRENT_HEADERS}) 
source_group("Source" FILES ${CURRENT_SOURCES}) 

add_library(ROIALIGN ${CURRENT_SOURCES})
target_link_libraries(ROIALIGN "${TORCH_LIBRARIES}")

target_include_directories(ROIALIGN PUBLIC ${PROJECT_SOURCE_DIR})
