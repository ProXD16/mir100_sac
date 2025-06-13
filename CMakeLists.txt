cmake_minimum_required(VERSION 3.0.2)
project(mir100_sac)
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  nav_msgs
  geometry_msgs
  tf
)
catkin_package()
include_directories(${catkin_INCLUDE_DIRS})