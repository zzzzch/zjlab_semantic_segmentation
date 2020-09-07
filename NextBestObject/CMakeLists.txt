add_executable(NextBestObject
        NextBestObject.cpp)

target_include_directories(NextBestObject
        PUBLIC
        ${PCL_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/base
        ${CMAKE_CURRENT_SOURCE_DIR}
        )

target_link_libraries(NextBestObject
        PUBLIC
        ${PCL_LIBRARIES}
        #base_lib
        )




