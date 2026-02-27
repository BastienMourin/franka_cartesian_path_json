#include <rclcpp/rclcpp.hpp>
#include <rclcpp/parameter_client.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nlohmann/json.hpp>
#include <fstream>
#include <vector>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <Eigen/Dense>
#include <signal.h>
#include <atomic>

using json = nlohmann::json;

std::atomic<bool> g_shutdown_requested(false);
void signalHandler(int) { g_shutdown_requested = true; }

// ==========================================================
// Transform a PoseStamped from its frame_id -> target_frame via /tf
// ==========================================================
bool transformPose(
    const geometry_msgs::msg::PoseStamped &pose_in,
    const std::string                     &target_frame,
    geometry_msgs::msg::PoseStamped       &pose_out,
    tf2_ros::Buffer                       &tf_buffer,
    const rclcpp::Logger                  &logger)
{
    if (pose_in.header.frame_id == target_frame) {
        pose_out = pose_in;
        return true;
    }
    try {
        pose_out = tf_buffer.transform(pose_in, target_frame,
                                       tf2::durationFromSec(0.1));
        return true;
    } catch (const tf2::TransformException &ex) {
        RCLCPP_WARN(logger, "TF %s -> %s failed: %s",
                    pose_in.header.frame_id.c_str(),
                    target_frame.c_str(), ex.what());
        return false;
    }
}

// ==========================================================
// Compute quaternion from surface normal (expressed in frame_id)
// ==========================================================
geometry_msgs::msg::Quaternion
computeQuaternionFromSurfaceNormal(const json &normal_json)
{
    Eigen::Vector3d normal(normal_json[0], normal_json[1], normal_json[2]);
    normal.normalize();

    Eigen::Vector3d z_axis = -normal;
    Eigen::Vector3d x_axis(1, 0, 0);
    if (std::fabs(x_axis.dot(z_axis)) > 0.99)
        x_axis = Eigen::Vector3d(0, 1, 0);

    Eigen::Vector3d y_axis = z_axis.cross(x_axis).normalized();
    x_axis = y_axis.cross(z_axis).normalized();

    Eigen::Matrix3d R;
    R.col(0) = x_axis;
    R.col(1) = y_axis;
    R.col(2) = z_axis;

    Eigen::Quaterniond q(R);
    geometry_msgs::msg::Quaternion q_msg;
    q_msg.w = q.w();
    q_msg.x = q.x();
    q_msg.y = q.y();
    q_msg.z = q.z();
    return q_msg;
}

// ==========================================================
// MAIN
// ==========================================================
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = rclcpp::Node::make_shared("franka_cartesian_path_json_moveit");
    signal(SIGINT, signalHandler);

    // ------------------------------------------------------
    // 1️⃣  TF2 buffer + listener (subscribes to /tf and /tf_static)
    // ------------------------------------------------------
    auto tf_buffer   = std::make_shared<tf2_ros::Buffer>(node->get_clock());
    auto tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);

    RCLCPP_INFO(node->get_logger(), "Waiting 2 s for TF to become available...");
    rclcpp::sleep_for(std::chrono::seconds(2));

    // ------------------------------------------------------
    // 2️⃣  Fetch MoveIt parameters from /move_group
    // ------------------------------------------------------
    RCLCPP_INFO(node->get_logger(), "Waiting for move_group parameters...");

    auto param_client =
        std::make_shared<rclcpp::SyncParametersClient>(node, "move_group");

    while (!param_client->wait_for_service(std::chrono::seconds(1))) {
        if (!rclcpp::ok()) {
            RCLCPP_ERROR(node->get_logger(),
                         "Interrupted while waiting for move_group.");
            return 1;
        }
        RCLCPP_INFO(node->get_logger(), "Waiting for move_group parameter service...");
    }

    try {
        node->declare_parameter("robot_description",
            param_client->get_parameter<std::string>("robot_description"));
        node->declare_parameter("robot_description_semantic",
            param_client->get_parameter<std::string>("robot_description_semantic"));
    } catch (const std::exception &e) {
        RCLCPP_FATAL(node->get_logger(),
                     "Failed to fetch MoveIt parameters: %s", e.what());
        return 1;
    }

    RCLCPP_INFO(node->get_logger(), "MoveIt parameters loaded.");

    // ------------------------------------------------------
    // 3️⃣  Initialize MoveGroupInterface
    // ------------------------------------------------------
    moveit::planning_interface::MoveGroupInterface move_group(node, "fr3_arm");

    // MoveIt plans in the robot's planning frame (fr3_link0).
    // All poses must be expressed in this frame before being sent.
    const std::string target_frame = move_group.getPlanningFrame();
    RCLCPP_INFO(node->get_logger(),
                "MoveGroupInterface initialized. Planning frame: %s",
                target_frame.c_str());

    // ------------------------------------------------------
    // 4️⃣  Load JSON
    // ------------------------------------------------------
    std::string package_path =
        ament_index_cpp::get_package_share_directory("ignacio_cartesian_pose_json");

    node->declare_parameter("waypoints_file", "waypoints_test.json");
    std::string filename =
        package_path + "/test_gripper_traj/" +
        node->get_parameter("waypoints_file").as_string();

    std::ifstream file(filename);
    if (!file.is_open()) {
        RCLCPP_ERROR(node->get_logger(), "Cannot open JSON file: %s", filename.c_str());
        return 1;
    }
    json j;
    file >> j;

    double pause_after_start = 1.0;

    // ------------------------------------------------------
    // 5️⃣  Process handles
    // ------------------------------------------------------
    for (const auto &handle : j["handles"])
    {
        if (g_shutdown_requested) break;

        int handle_id = handle["handle_id"].get<int>();

        std::string frame_id = target_frame;
        if (handle.contains("frame_id"))
            frame_id = handle["frame_id"].get<std::string>();

        // -- Initial grasp pose --
        geometry_msgs::msg::PoseStamped grasp_in;
        grasp_in.header.frame_id = frame_id;
        grasp_in.header.stamp    = node->get_clock()->now();
        grasp_in.pose.position.x = handle["grasp_position"][0];
        grasp_in.pose.position.y = handle["grasp_position"][1];
        grasp_in.pose.position.z = handle["grasp_position"][2];
        grasp_in.pose.orientation = computeQuaternionFromSurfaceNormal(handle["surface_normal"]);

        geometry_msgs::msg::PoseStamped grasp_out;
        if (!transformPose(grasp_in, target_frame, grasp_out, *tf_buffer, node->get_logger())) {
            RCLCPP_ERROR(node->get_logger(),
                         "Cannot transform grasp pose for handle %d — skipping.", handle_id);
            continue;
        }

        moveit::planning_interface::MoveGroupInterface::Plan start_plan;
        move_group.setPoseTarget(grasp_out.pose);
        if (move_group.plan(start_plan) !=
            moveit::planning_interface::MoveItErrorCode::SUCCESS)
        {
            RCLCPP_ERROR(node->get_logger(),
                         "Failed to plan to start pose of handle %d", handle_id);
            continue;
        }

        move_group.execute(start_plan);
        rclcpp::sleep_for(
            std::chrono::milliseconds(static_cast<int>(pause_after_start * 1000)));

        // -- Waypoints --
        std::vector<geometry_msgs::msg::Pose> waypoints;

        for (const auto &wp : handle["waypoints"])
        {
            if (g_shutdown_requested) break;

            geometry_msgs::msg::PoseStamped wp_in;
            wp_in.header.frame_id = frame_id;
            wp_in.header.stamp    = node->get_clock()->now();
            wp_in.pose.position.x    = wp["position"][0];
            wp_in.pose.position.y    = wp["position"][1];
            wp_in.pose.position.z    = wp["position"][2];
            wp_in.pose.orientation.w = wp["quaternion_wxyz"][0];
            wp_in.pose.orientation.x = wp["quaternion_wxyz"][1];
            wp_in.pose.orientation.y = wp["quaternion_wxyz"][2];
            wp_in.pose.orientation.z = wp["quaternion_wxyz"][3];

            geometry_msgs::msg::PoseStamped wp_out;
            if (transformPose(wp_in, target_frame, wp_out, *tf_buffer, node->get_logger()))
                waypoints.push_back(wp_out.pose);
            else
                RCLCPP_WARN(node->get_logger(),
                            "Skipping waypoint t=%.2f (TF unavailable).",
                            wp["t"].get<double>());
        }

        if (waypoints.empty()) {
            RCLCPP_WARN(node->get_logger(),
                        "No valid waypoints for handle %d — skipping trajectory.", handle_id);
            continue;
        }

        moveit_msgs::msg::RobotTrajectory trajectory;
        double fraction =
            move_group.computeCartesianPath(waypoints, 0.01, 0.0, trajectory);

        if (fraction > 0.99) {
            move_group.execute(trajectory);
            RCLCPP_INFO(node->get_logger(),
                        "Executed Cartesian trajectory for handle %d (%.1f%%).",
                        handle_id, fraction * 100.0);
        } else {
            RCLCPP_WARN(node->get_logger(),
                        "Cartesian path incomplete for handle %d (%.1f%%) — skipping.",
                        handle_id, fraction * 100.0);
        }
    }

    RCLCPP_INFO(node->get_logger(), "All handles processed.");
    rclcpp::shutdown();
    return 0;
}