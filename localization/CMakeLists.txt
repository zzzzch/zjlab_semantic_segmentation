project(localization)

link_directories(
        ${Boost_LIBRARY_DIRS}
        ${CMAKE_SOURCE_DIR}/base/include
)


add_executable(localization
        rgb_localization.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/base/src
        ${CMAKE_SOURCE_DIR}/base/include
        )

target_include_directories(localization
        PUBLIC
        ${CMAKE_SOURCE_DIR}/base/include
        ${PCL_INCLUDE_DIRS}
        ${EIGEN_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${OPENCV_INCLUDE_DIRS}
        ${OPENGL_INCLUDE_DIR}
        ${OpenMesh_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${GLOG_INCLUDE_DIR}
        ${GFLAGS_INCLUDE_DIR}
        )


target_link_libraries(localization
        PUBLIC
        base
        ${PCL_LIBRARIES}
        ${Eigen_LIBRARIES}
        ${OpenCV_LIBS}
        ${OPENGL_LIBRARIES}
        ${OpenMesh_LIBS}
        ${GLOG_LIBRARY}
        ${gflags_LIBRARIES}
        )

add_executable(vote_hough
        pose_estimation.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/base/src
        ${CMAKE_SOURCE_DIR}/base/include
        )

target_include_directories(vote_hough
        PUBLIC
        ${CMAKE_SOURCE_DIR}/base/include
        ${PCL_INCLUDE_DIRS}
        ${EIGEN_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${OPENCV_INCLUDE_DIRS}
        ${OPENGL_INCLUDE_DIR}
        ${OpenMesh_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${GLOG_INCLUDE_DIR}
        ${GFLAGS_INCLUDE_DIR}
        )

target_link_libraries(vote_hough
        PUBLIC
        base
        ${PCL_LIBRARIES}
        ${Eigen_LIBRARIES}
        ${OpenCV_LIBS}
        ${OPENGL_LIBRARIES}
        ${OpenMesh_LIBS}
        ${GLOG_LIBRARY}
        ${gflags_LIBRARIES}
        )



add_executable(gl_show
        gl_show.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/base/src
        ${CMAKE_SOURCE_DIR}/base/include
        )

target_include_directories(gl_show
        PUBLIC
        ${CMAKE_SOURCE_DIR}/base/include
        ${PCL_INCLUDE_DIRS}
        ${EIGEN_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${OPENCV_INCLUDE_DIRS}
        ${OPENGL_INCLUDE_DIR}
        ${GLUT_INCLUDE_DIR}
        ${OpenMesh_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${GLOG_INCLUDE_DIR}
        ${GFLAGS_INCLUDE_DIR}
        )

target_link_libraries(gl_show
        PUBLIC
        base
        ${PCL_LIBRARIES}
        ${Eigen_LIBRARIES}
        ${OpenCV_LIBS}
        ${GLUT_LIBRARIES}
        ${OPENGL_LIBRARIES}
        ${OpenMesh_LIBS}
        ${GLOG_LIBRARY}
        ${gflags_LIBRARIES}
        )
