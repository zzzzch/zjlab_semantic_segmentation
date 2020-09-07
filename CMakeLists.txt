cmake_minimum_required(VERSION 3.9)
project(untitled1)

set(CMAKE_CXX_STANDARD 17)

find_package(PCL 1.9 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})

add_executable(untitled1 main.cpp)

target_link_libraries (untitled1
        ${PCL_LIBRARIES})

add_subdirectory(NextBestObject)
#add_subdirectory(base)
