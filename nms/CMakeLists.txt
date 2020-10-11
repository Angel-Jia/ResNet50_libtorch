cmake_minimum_required(VERSION 3.0)

project(NMS LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 14)
# set(CMAKE_VERBOSE_MAKEFILE on)

find_package(Torch REQUIRED)

file(GLOB_RECURSE CURRENT_HEADERS  *.h *.hpp *.cuh)
file(GLOB CURRENT_SOURCES  *.cpp *.cu)

source_group("Include" FILES ${CURRENT_HEADERS}) 
source_group("Source" FILES ${CURRENT_SOURCES}) 

add_library(NMS ${CURRENT_SOURCES})
target_link_libraries(NMS "${TORCH_LIBRARIES}")
target_include_directories(NMS PUBLIC ${PROJECT_SOURCE_DIR})
