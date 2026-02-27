cmake_minimum_required(VERSION 3.5)
project(ignacio_cartesian_pose_json)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# ===========================================================================
# Dependencies
# ===========================================================================
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)

# ===========================================================================
# Include directories
# ===========================================================================
include_directories(include)

# ===========================================================================
# Executables
# ===========================================================================
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

add_executable(franka_cartesian_path_json_dual_increments
  src/franka_cartesian_path_json_dual_increments.cpp
)

# ===========================================================================
# Target dependencies - Core ROS packages (used by all executables)
# ===========================================================================
foreach(target 
  franka_cartesian_path_json_moveit
  franka_cartesian_path_json
  franka_cartesian_path_json_dual
  franka_cartesian_path_json_increments
  franka_cartesian_path_json_dual_increments
)
  ament_target_dependencies(${target}
    rclcpp
    geometry_msgs
    tf2_ros
    tf2_geometry_msgs
  )
  target_link_libraries(${target}
    yaml-cpp
    Eigen3::Eigen
  )
endforeach()

# ===========================================================================
# Target dependencies - Specialized packages
# ===========================================================================
# MoveIt (only for moveit variant)
ament_target_dependencies(franka_cartesian_path_json_moveit
  moveit_ros_planning_interface
)

# Visualization markers (only for dual_increments variant)
ament_target_dependencies(franka_cartesian_path_json_dual_increments
  visualization_msgs
)

# ===========================================================================
# Install
# ===========================================================================
install(TARGETS 
  franka_cartesian_path_json_moveit
  franka_cartesian_path_json
  franka_cartesian_path_json_dual
  franka_cartesian_path_json_increments
  franka_cartesian_path_json_dual_increments
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

install(DIRECTORY test_gripper_traj/
  DESTINATION share/${PROJECT_NAME}/test_gripper_traj
  FILES_MATCHING PATTERN "*.json"
)

install(DIRECTORY calibration_results/
  DESTINATION share/${PROJECT_NAME}/calibration_results
)

install(PROGRAMS
  scripts/wrench_logger_node.py
  scripts/obs_recorder_node.py
  scripts/realsense_recorder_node.py
  scripts/recording_manager_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Rename Python scripts (remove .py extension)
set_property(INSTALL "lib/${PROJECT_NAME}/wrench_logger_node.py"
  PROPERTY RENAME "wrench_logger_node"
)
set_property(INSTALL "lib/${PROJECT_NAME}/obs_recorder_node.py"
  PROPERTY RENAME "obs_recorder_node"
)
set_property(INSTALL "lib/${PROJECT_NAME}/realsense_recorder_node.py"
  PROPERTY RENAME "realsense_recorder_node"
)
set_property(INSTALL "lib/${PROJECT_NAME}/recording_manager_node.py"
  PROPERTY RENAME "recording_manager_node"
)

ament_package()