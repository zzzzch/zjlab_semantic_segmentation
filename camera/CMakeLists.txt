project(camera)

set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -lgxiapi -pthread" )

add_executable(camera
        main.cpp
        GxIAPI.h
        DxImageProc.h
        )

set(XIAPI_INCLUDE /usr/lib/)
set(XIAPI_LIBS /usr/lib/libgxiapi.so)

message(${XIAPI_INCLUDE})
target_include_directories(camera
        PUBLIC
        ${XIAPI_INCLUDE}
        ${OPENCV_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
        )

target_link_libraries(camera
        PUBLIC
        ${XIAPI_LIBS}
        ${OpenCV_LIBS}
        )



