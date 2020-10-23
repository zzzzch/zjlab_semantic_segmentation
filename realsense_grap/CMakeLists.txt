project(realsense_grap)

link_directories(
        ${Boost_LIBRARY_DIRS}
        ${CMAKE_SOURCE_DIR}/base/include
        ${CMAKE_SOURCE_DIR}/imgui

)

add_executable(realsense_grap
        src/realsense_grap.cpp
        include/realsense_base.h
        ${CMAKE_SOURCE_DIR}/base/src
        ${CMAKE_SOURCE_DIR}/base/include
        ${CMAKE_SOURCE_DIR}/imgui
        )

set(GLFW_INCLUDE_DIR /usr/local/include/)
set(GLFW_LIBRARIES /usr/lib/x86_64-linux-gnu/libglfw.so.3)

target_include_directories(realsense_grap
        PUBLIC
        ${CMAKE_SOURCE_DIR}/base/include
        ${CMAKE_SOURCE_DIR}/imgui
        ${PCL_INCLUDE_DIRS}
        ${EIGEN_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${OPENCV_INCLUDE_DIRS}
        ${OPENGL_INCLUDE_DIR}
        ${GLEW_INCLUDE_DIR}
        ${GLFW_INCLUDE_DIR}
        ${realsense2_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${GLOG_INCLUDE_DIR}
        ${GFLAGS_INCLUDE_DIR}
        )

target_link_libraries(realsense_grap
        PUBLIC
        base
        imgui
        ${PCL_LIBRARIES}
        ${Eigen_LIBRARIES}
        ${OpenCV_LIBS}
        ${OPENGL_LIBRARIES}
        ${GLEW_LIBRARIES}
        ${GLFW_LIBRARIES}
        ${realsense2_LIBRARY}
        ${GLOG_LIBRARY}
        ${gflags_LIBRARIES}
        )

