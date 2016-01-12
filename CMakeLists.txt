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
  src/external/5point
  ${catkin_INCLUDE_DIRS}
  ${Eigen_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS})

add_executable(stereo_odometry
  src/node.cpp
  src/featools.cpp
  src/feature.cpp
  src/frame.cpp
  src/tracker.cpp
  src/optimizer.cpp
  src/external/5point/5point.cpp
  src/external/5point/Rpoly.cpp)
target_link_libraries(stereo_odometry
  ${eigen_LIBRARIES}
  ${CERES_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${catkin_LIBRARIES})