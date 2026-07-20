cmake_minimum_required(VERSION 3.0.2)
project(aplvio)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

option(ENABLE_CUDA "Include NVidia Cuda Runtime support" OFF)

if (ENABLE_CUDA)
  set(ENABLE_CUDA 1)
  message(STATUS "catkin with CUDA")
else()
  set(ENABLE_CUDA 0)
  message(STATUS "catkin without CUDA")
endif ()

configure_file(
  "${PROJECT_SOURCE_DIR}/settings/catkin_settings.h.in"
  "${PROJECT_SOURCE_DIR}/settings/catkin_settings.h"
)

list(APPEND CMAKE_INCLUDE_PATH 
  "${CMAKE_CURRENT_LIST_DIR}/settings"
  "${CMAKE_CURRENT_LIST_DIR}/include/visual/line_descriptor/include/"
  "${CMAKE_CURRENT_LIST_DIR}/include/visual/line_descriptor/src/"
)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  std_msgs
  roscpp
  rospy
  tf
  cv_bridge
  camera_model
)

find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)
# set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake/opencv4")
# set(OpenCV_DIR "/usr/local/lib/cmake/opencv")
find_package(OpenCV REQUIRED)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES aplvio
#  CATKIN_DEPENDS geometry_msgs roscpp rospy std_msgs
#  DEPENDS system_lib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${CMAKE_INCLUDE_PATH}
)

# file(GLOB_RECURSE PROJECT_SOURCE_FILES ${CMAKE_CURRENT_LIST_DIR}/src/*/*.cpp)
aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/src/estimator ESTIMATOR_DIR)
aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/src/factor FACTOR_DIR)
aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/src/inertial INERTIAL_DIR)
aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/src/initializer INITIALIZER_DIR)
aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/src/rosMaster ROSMASTER_DIR)
aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/src/visual VISUAL_DIR)

add_library(${PROJECT_NAME}_lib
  ${ESTIMATOR_DIR}
  ${FACTOR_DIR}
  ${INERTIAL_DIR}
  ${INITIALIZER_DIR}
  ${ROSMASTER_DIR}
  ${VISUAL_DIR}
)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/include/visual/line_descriptor)

target_link_libraries(${PROJECT_NAME}_lib
  ${catkin_LIBRARIES}
  ${CERES_LIBRARIES}
  ${OpenCV_LIBS}
  line_descriptor # line feature
)

add_executable(${PROJECT_NAME}_ros_node 
  src/aplvio_ros_node.cpp
)

target_link_libraries(${PROJECT_NAME}_ros_node
  ${PROJECT_NAME}_lib
)

add_executable(${PROJECT_NAME}_kitti_node 
  src/aplvio_kitti_node.cpp
)

target_link_libraries(${PROJECT_NAME}_kitti_node
  ${PROJECT_NAME}_lib
)