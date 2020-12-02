project(cuda C CXX)

file(GLOB_RECURSE ${PROJECT_NAME}_files src/*.cu)
## cuda
find_package(CUDA 9.0 QUIET REQUIRED)

message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
set_directory_properties(PROPERTIES COMPILE_DEFINITIONS "")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    list(APPEND CUDA_NVCC_FLAGS -g -G)
endif ()
list(APPEND CUDA_NVCC_FLAGS -std=c++11 -gencode arch=compute_61,code=sm_61) # TODO compute capacity on server??

CUDA_ADD_LIBRARY(${PROJECT_NAME} ${${PROJECT_NAME}_files})

target_include_directories(${PROJECT_NAME} PRIVATE
        PUBLIC
        ${CUDA_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        )

target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBS}
        )