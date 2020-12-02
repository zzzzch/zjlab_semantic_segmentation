         cmake_minimum_required(VERSION 3.10.0)
         project(test_gpu)

         set(CMAKE_CXX_STANDARD 11)

         find_package(OpenMP)
         if (OPENMP_FOUND)
             set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
             set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
         endif()

         set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-10.0")

         find_package(CUDA REQUIRED)
         if(CUDA_FOUND)
#             message("cuda include is " ${CUDA_INCLUDE_DIRS})
#             message("cuda lib is " ${CUDA_LIBRARIES})
             include_directories(${CUDA_INCLUDE_DIRS}  ${Boost_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})
             link_directories(${CUDA_LIBRARIES}   ${CBoost_LIBRARIES} ${PCL_LIBRARIES})
         endif()


#         get_directory_property(dir_defs DIRECTORY ${PROJECT_SOURCE_DIR} COMPILE_DEFINITIONS)
#         set(vtk_flags)
#         foreach(it ${dir_defs})
#             if(it MATCHES "vtk*")
#                 list(APPEND vtk_flags ${it})
#             endif()
#         endforeach()
#
#         foreach(d ${vtk_flags})
#             remove_definitions(-D${d})
#         endforeach()


#         set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--disable-warnings;--ptxas-options=-v;-use_fast_math;-lineinfo;-std=c++11;--expt-relaxed-constexpr)

         cuda_add_executable(test_gpu
                 src/test_main.cu)

         add_executable(test_main
                 src/test_no_gpu.cpp)

         #         target_include_directories(test_gpu PRIVATE
##                 ${PCL_INCLUDE_DIRS}
##                 ${Eigen3_INCLUDE_DIRS}
#
#                 )
#         target_link_libraries(test_gpu
##                 ${PCL_LIBRARIES}
##                 ${Eigen3_LIBRARIES}
#                 )



         #
#         add_definitions(-DUSE_VGICP_CUDA)
#
#         cuda_add_library(fast_vgicp_cuda
#                 src/fast_gicp/cuda/fast_vgicp_cuda.cu
#                 )
#         target_include_directories(fast_vgicp_cuda PRIVATE
#                 include
#                 thirdparty/Sophus
#                 thirdparty/Eigen
#                 thirdparty/nvbio
#                 )
#         cuda_add_cublas_to_target(fast_vgicp_cuda)
#         #   target_link_libraries(fast_vgicp_cuda
#         #     cublas
#         #   )
#
#
#         # add vgicp_cuda to libfast_gicp
#         target_sources(fast_gicp PRIVATE
#                 src/fast_gicp/gicp/fast_vgicp_cuda.cpp
#                 )
#         target_link_libraries(fast_gicp
#                 fast_vgicp_cuda
#                 )
#         add_dependencies(fast_gicp fast_vgicp_cuda)
