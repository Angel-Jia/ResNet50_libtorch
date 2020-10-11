cmake_minimum_required(VERSION 3.17)

project(FASTER)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_VERBOSE_MAKEFILE on)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

# include_directories(${PROJECT_SOURCE_DIR}/nms)
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_subdirectory(nms)
add_subdirectory(roialign)
add_subdirectory(nlohmann_json)
set(JSON_BuildTests OFF CACHE INTERNAL "")

target_compile_options(ROIALIGN PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       -D__CUDA_NO_HALF_OPERATORS__
                       >)
add_executable(${PROJECT_NAME} test.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES} ${OpenCV_LIBS} NMS ROIALIGN nlohmann_json::nlohmann_json)
