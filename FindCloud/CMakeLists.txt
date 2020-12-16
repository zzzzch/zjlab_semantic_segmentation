project(FindCloud)

#find_package(util REQUIRED)
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

#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xlinker -export-dynamic")

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -export-dynamic")

set(LIBUTIL_LIBS /usr/lib/x86_64-linux-gnu/libutil.so)

add_executable(charging_gun_detect
        charging_gun_detect.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/base/src
        ${CMAKE_SOURCE_DIR}/base/include
        )

target_include_directories(charging_gun_detect
        PUBLIC
        ${CMAKE_SOURCE_DIR}/base/include
        ${Python3_INCLUDE_DIRS}
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
        ${Python3_LIBRARIES}
        ${LIBUTIL_LIBS}
        #        ${PCL_LIBRARIES}
        ${Eigen_LIBRARIES}
        ${OpenCV_LIBS}
#        ${OPENGL_LIBRARIES}
        ${GLOG_LIBRARY}
        ${gflags_LIBRARIES}
        )

add_executable(stereo_sgbm
        stereo_sgbm.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/base/include
        )

target_include_directories(stereo_sgbm
        PUBLIC
        ${CMAKE_SOURCE_DIR}/base/include
        ${Python3_INCLUDE_DIRS}
        #        ${PCL_INCLUDE_DIRS}
        ${EIGEN_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${OPENCV_INCLUDE_DIRS}
        #        ${OPENGL_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${GLOG_INCLUDE_DIR}
        ${GFLAGS_INCLUDE_DIR}
        )

target_link_libraries(stereo_sgbm
        PUBLIC
        base
        ${Python3_LIBRARIES}
        ${LIBUTIL_LIBS}
        #        ${PCL_LIBRARIES}
        ${Eigen_LIBRARIES}
        ${OpenCV_LIBS}
        #        ${OPENGL_LIBRARIES}
        ${GLOG_LIBRARY}
        ${gflags_LIBRARIES}
        )

add_executable(sgm_match_stereo
        sgm_match_stereo.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        src/sgm_util.cpp
        src/SemiGlobalMatching.cpp
        ${CMAKE_SOURCE_DIR}/base/include
        )

target_include_directories(sgm_match_stereo
        PUBLIC
        ${CMAKE_SOURCE_DIR}/base/include
        ${Python3_INCLUDE_DIRS}
        #        ${PCL_INCLUDE_DIRS}
        ${EIGEN_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${OPENCV_INCLUDE_DIRS}
        #        ${OPENGL_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${GLOG_INCLUDE_DIR}
        ${GFLAGS_INCLUDE_DIR}
        )

target_link_libraries(sgm_match_stereo
        PUBLIC
        base
        ${Python3_LIBRARIES}
        ${LIBUTIL_LIBS}
        #        ${PCL_LIBRARIES}
        ${Eigen_LIBRARIES}
        ${OpenCV_LIBS}
        #        ${OPENGL_LIBRARIES}
        ${GLOG_LIBRARY}
        ${gflags_LIBRARIES}
        )

add_executable(detect_image
        detect_image.cpp
        ${CMAKE_SOURCE_DIR}/base/src
        ${CMAKE_SOURCE_DIR}/base/include
        )

target_include_directories(detect_image
        PUBLIC
        ${CMAKE_SOURCE_DIR}/base/include
        ${Python3_INCLUDE_DIRS}
        ${EIGEN_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${OPENCV_INCLUDE_DIRS}
        ${GLOG_INCLUDE_DIR}
        ${GFLAGS_INCLUDE_DIR}
        )

target_link_libraries(detect_image
        PUBLIC
        base
        ${Python3_LIBRARIES}
        ${LIBUTIL_LIBS}
        ${Eigen_LIBRARIES}
        ${OpenCV_LIBS}
        ${GLOG_LIBRARY}
        ${gflags_LIBRARIES}
        )
