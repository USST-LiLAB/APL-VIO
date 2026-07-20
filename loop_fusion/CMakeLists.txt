cmake_minimum_required(VERSION 3.0.2)
project(loop_fusion)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
	roscpp
	std_msgs
	nav_msgs
	camera_model
	cv_bridge
	roslib
)

find_package(Ceres REQUIRED)

# set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake/opencv4")
# set(OpenCV_DIR "/usr/local/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)

include_directories(${catkin_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS}  ${EIGEN3_INCLUDE_DIR})

catkin_package()

add_executable(loop_fusion_node
    src/pose_graph_node.cpp
    src/pose_graph.cpp
    src/keyframe.cpp
    src/utility/CameraPoseVisualization.cpp
    src/ThirdParty/DBoW/BowVector.cpp
    src/ThirdParty/DBoW/FBrief.cpp
    src/ThirdParty/DBoW/FeatureVector.cpp
    src/ThirdParty/DBoW/QueryResults.cpp
    src/ThirdParty/DBoW/ScoringObject.cpp
    src/ThirdParty/DUtils/Random.cpp
    src/ThirdParty/DUtils/Timestamp.cpp
    src/ThirdParty/DVision/BRIEF.cpp
    src/ThirdParty/VocabularyBinary.cpp
    )

target_link_libraries(loop_fusion_node 
	${catkin_LIBRARIES} 
	${OpenCV_LIBS} 
	${CERES_LIBRARIES}
) 
