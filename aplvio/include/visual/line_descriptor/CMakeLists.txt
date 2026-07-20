cmake_minimum_required(VERSION 3.0.2)
project(line_descriptor)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fPIC")

# set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake/opencv4")
# set(OpenCV_DIR "/usr/local/lib/cmake/opencv")
find_package(OpenCV REQUIRED)

include_directories(
  include
)

file(GLOB_RECURSE PROJECT_SOURCE_FILES ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)

add_library(${PROJECT_NAME} SHARED
  ${PROJECT_SOURCE_FILES}
)

target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBS}
)
