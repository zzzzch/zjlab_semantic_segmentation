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



add_executable(charging_gun_detect
        charging_gun_detect.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/base/src
        ${CMAKE_SOURCE_DIR}/base/include
        )

target_include_directories(charging_gun_detect
        PUBLIC
        ${CMAKE_SOURCE_DIR}/base/include
#        ${PCL_INCLUDE_DIRS}
        ${EIGEN_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${OPENCV_INCLUDE_DIRS}
#        ${OPENGL_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${GLOG_INCLUDE_DIR}
        ${GFLAGS_INCLUDE_DIR}
        )

target_link_libraries(charging_gun_detect
        PUBLIC
        base
#        ${PCL_LIBRARIES}
        ${Eigen_LIBRARIES}
        ${OpenCV_LIBS}
#        ${OPENGL_LIBRARIES}
        ${GLOG_LIBRARY}
        ${gflags_LIBRARIES}
        )

