cmake_minimum_required(VERSION 3.9)
project(untitled1)

set(CMAKE_CXX_STANDARD 17)


find_package(PCL 1.9 REQUIRED)
find_package(Boost REQUIRED QUIET)
find_package(OpenCV REQUIRED)

## glog
find_package(Glog REQUIRED)
if(NOT GLOG_INCLUDE_DIR)
    set(GLOG_INCLUDE_DIR /usr/lib/x86_64-linux-gnu/)
    set(GLOG_LIBRARY /usr/local/lib/libglog.so)
endif(NOT GLOG_INCLUDE_DIR)


## gflags
find_package(Gflags REQUIRED)
if(NOT GFLAGS_INCLUDE_DIR)
    message("Gflags path is wrong , can't find it" )
endif(NOT GFLAGS_INCLUDE_DIR)

# opengl
find_package(OpenGL REQUIRED)
if(NOT (OPENGL_INCLUDE_DIR OR OPENGL_LIBRARIES))
    message("opengl path is wrong , can't find it" )
    message("opengl include is " ${OPENGL_INCLUDE_DIR})
    message("opengl lab is " ${OPENGL_LIBRARIES})
endif(NOT (OPENGL_INCLUDE_DIR OR OPENGL_LIBRARIES))

# glut
find_package(GLUT REQUIRED)
if(NOT (GLUT_INCLUDE_DIR OR GLUT_LIBRARIES))
    message("glut path is wrong , can't find it" )
    message("glut include is " ${GLUT_INCLUDE_DIR})
    message("glut lab is " ${GLUT_LIBRARIES})
endif(NOT (GLUT_INCLUDE_DIR OR GLUT_LIBRARIES))

#/usr/lib/x86_64-linux-gnu/libglut.so

## openmesh
find_package(OpenMesh REQUIRED)
#if(NOT (OpenMesh_INCLUDE_DIRS AND OpenMesh_LIBS))
#    message("openmash path is wrong , can't find it" )
#    message("mesh include is " ${OpenMesh_INCLUDE_DIRS})
#    message("mesh libs is " ${OpenMesh_LIBS})
#
#endif(NOT (OpenMesh_INCLUDE_DIRS AND OpenMesh_LIBS))

include_directories(${PCL_INCLUDE_DIRS}
        ${GLOG_INCLUDE_DIR}
        ${GFLAGS_INCLUDE_DIR}
        ${EIGEN_INCLUDE_DIRS}
        )

#add_executable(untitled1 main.cpp)
#
#target_link_libraries (untitled1
#        ${GLOG_LIBRARY}
#        ${gflags_LIBRARIES}
#        ${Eigen_LIBRARIES}
#        ${PCL_LIBRARIES})

add_subdirectory(base)
add_subdirectory(FindCloud)
add_subdirectory(localization)
#add_subdirectory(camera)

