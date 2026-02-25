cmake_minimum_required(VERSION 3.5)
project(ignacio_cartesian_pose_json)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

# Include directories
include_directories(
  include
)

# Executables
add_executable(franka_cartesian_path_json_moveit
  src/franka_cartesian_path_json_moveit.cpp
)

add_executable(franka_cartesian_path_json
  src/franka_cartesian_path_json.cpp
)

add_executable(franka_cartesian_path_json_dual
  src/franka_cartesian_path_json_dual.cpp
)

add_executable(franka_cartesian_path_json_increments
  src/franka_cartesian_path_json_increments.cpp
)

# Target dependencies for franka_cartesian_path_json_moveit
ament_target_dependencies(franka_cartesian_path_json_moveit
  rclcpp
  geometry_msgs
  tf2_ros
  moveit_ros_planning_interface
)

target_link_libraries(franka_cartesian_path_json_moveit
  yaml-cpp
  Eigen3::Eigen
)

# Target dependencies for franka_cartesian_path_json
ament_target_dependencies(franka_cartesian_path_json
  rclcpp
  geometry_msgs
  tf2_ros
  moveit_ros_planning_interface
)

target_link_libraries(franka_cartesian_path_json
  yaml-cpp
  Eigen3::Eigen
)

# Target dependencies for franka_cartesian_path_json_dual
ament_target_dependencies(franka_cartesian_path_json_dual
  rclcpp
  geometry_msgs
  tf2_ros
  moveit_ros_planning_interface
)

target_link_libraries(franka_cartesian_path_json_dual
  yaml-cpp
  Eigen3::Eigen
)

# Target dependencies for franka_cartesian_path_json_increments
ament_target_dependencies(franka_cartesian_path_json_increments
  rclcpp
  geometry_msgs
  tf2_ros
  moveit_ros_planning_interface
)

target_link_libraries(franka_cartesian_path_json_increments
  yaml-cpp
  Eigen3::Eigen
)


# Install executables
install(TARGETS franka_cartesian_path_json_moveit franka_cartesian_path_json franka_cartesian_path_json_dual franka_cartesian_path_json_increments
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

# Install gripper_traj folder
install(DIRECTORY gripper_traj/
  DESTINATION share/${PROJECT_NAME}/gripper_traj
  FILES_MATCHING PATTERN "*.json"
)

# Install calibration_results folder
install(DIRECTORY calibration_results/
  DESTINATION share/${PROJECT_NAME}/calibration_results
)

ament_package()