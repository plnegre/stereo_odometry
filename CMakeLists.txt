cmake_minimum_required(VERSION 2.8.3)

project(stereo_odometry)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  nav_msgs
  tf
  sensor_msgs
  image_transport
  message_filters
  cv_bridge
  image_geometry
  cmake_modules)

# Dependencies
find_package(PkgConfig)

# Dependencies - Eigen:
find_package(Eigen REQUIRED)

# Dependencies - OpenCV:
find_package(OpenCV REQUIRED)
link_directories(${OpenCV_LIBRARY_DIRS})

# Dependencies - Ceres:
find_package(Ceres REQUIRED)

catkin_package()

include_directories(include
  ${catkin_INCLUDE_DIRS}
  ${Eigen_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS})

add_executable(stereo_odometry
  src/node.cpp
  src/feature.cpp
  src/mappoint.cpp
  src/map.cpp
  src/frame.cpp
  src/tracking.cpp
  src/optimizer.cpp)
target_link_libraries(stereo_odometry
  ${eigen_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${CERES_LIBRARIES}
  ${catkin_LIBRARIES})