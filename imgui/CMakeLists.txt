project(imgui)

file(GLOB_RECURSE ${PROJECT_NAME}_CPP_FILES *.cpp *.c *.h *.hpp)

add_library(${PROJECT_NAME} ${${PROJECT_NAME}_CPP_FILES})

target_include_directories(${PROJECT_NAME}
        PUBLIC
#        ${CMAKE_CURRENT_SOURCE_DIR}/include
#        ${PCL_INCLUDE_DIRS}
#        ${EIGEN_INCLUDE_DIRS}
#        ${Boost_INCLUDE_DIRS}
#        ${OPENCV_INCLUDE_DIRS}
#        ${OPENGL_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
#        ${GLOG_INCLUDE_DIR}
#        ${GFLAGS_INCLUDE_DIR}
        )

target_link_libraries(${PROJECT_NAME}
        PUBLIC
#        ${PCL_LIBRARIES}
#        ${Eigen_LIBRARIES}
#        ${OpenCV_LIBS}
#        ${OPENGL_LIBRARIES}
#        ${GLOG_LIBRARY}
#        ${gflags_LIBRARIES}
#        PRIVATE nlohmann_json::nlohmann_json
        )

