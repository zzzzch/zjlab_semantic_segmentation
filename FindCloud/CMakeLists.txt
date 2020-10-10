project(FindCloud)

add_executable(FindCloud
        readply.cpp)

target_include_directories(FindCloud
        PUBLIC
        ${PCL_INCLUDE_DIRS}
        ${EIGEN_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/base/include
        ${GLOG_INCLUDE_DIR}
        ${GFLAGS_INCLUDE_DIR}
        )


link_directories(
        ${Boost_LIBRARY_DIRS}
        ${CMAKE_SOURCE_DIR}/base/include
)

target_link_libraries(FindCloud
        PUBLIC
        base
        ${PCL_LIBRARIES}
        ${Eigen_LIBRARIES}
        ${OpenCV_LIBS}
        ${GLOG_LIBRARY}
        ${gflags_LIBRARIES}
        )




