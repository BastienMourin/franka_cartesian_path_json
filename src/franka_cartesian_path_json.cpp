#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <nlohmann/json.hpp>
#include <fstream>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <Eigen/Dense>
#include <atomic>
#include <signal.h>

using json = nlohmann::json;

std::atomic<bool> g_shutdown_requested(false);
void signalHandler(int) { g_shutdown_requested = true; }

// ==========================================================
// Transform a PoseStamped from its frame_id -> target_frame via /tf
// Returns false (and logs a warning) if the transform is unavailable.
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
    auto node = rclcpp::Node::make_shared("franka_cartesian_path_json");
    signal(SIGINT, signalHandler);

    // ------------------------------------------------------
    // 1️⃣  TF2 buffer + listener (subscribes to /tf and /tf_static)
    // ------------------------------------------------------
    auto tf_buffer   = std::make_shared<tf2_ros::Buffer>(node->get_clock());
    auto tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);

    // Wait for TF tree to populate before querying
    RCLCPP_INFO(node->get_logger(), "Waiting 2 s for TF to become available...");
    rclcpp::sleep_for(std::chrono::seconds(2));

    // ------------------------------------------------------
    // 2️⃣  Publisher
    // ------------------------------------------------------
    auto pose_pub = node->create_publisher<geometry_msgs::msg::PoseStamped>(
        "/my_cartesian_impedance_controller/target_pose", 10);
    RCLCPP_INFO(node->get_logger(), "Pose publisher initialized.");

    // ------------------------------------------------------
    // 3️⃣  Load JSON
    // ------------------------------------------------------
    std::string package_path =
        ament_index_cpp::get_package_share_directory("ignacio_cartesian_pose_json");

    node->declare_parameter("waypoints_file", "waypoints_test_camera.json");
    std::string filename =
        package_path + "/gripper_traj/" +
        node->get_parameter("waypoints_file").as_string();

    std::ifstream file(filename);
    if (!file.is_open()) {
        RCLCPP_ERROR(node->get_logger(), "Cannot open JSON file: %s", filename.c_str());
        return 1;
    }
    json j;
    file >> j;

    // ------------------------------------------------------
    // 4️⃣  Process handles and publish poses
    // ------------------------------------------------------
    // All poses are transformed to fr3_link0 via live /tf before publishing.
    const std::string target_frame = "fr3_link0";
    double last_time = 0.0;

    for (const auto &handle : j["handles"])
    {
        if (g_shutdown_requested) break;

        int handle_id = handle["handle_id"].get<int>();

        std::string frame_id = "fr3_link0";
        if (handle.contains("frame_id"))
            frame_id = handle["frame_id"].get<std::string>();

        // -- Initial grasp pose --
        geometry_msgs::msg::PoseStamped pose_in;
        pose_in.header.frame_id = frame_id;
        pose_in.header.stamp    = node->get_clock()->now();

        pose_in.pose.position.x = handle["grasp_position"][0];
        pose_in.pose.position.y = handle["grasp_position"][1];
        pose_in.pose.position.z = handle["grasp_position"][2];
        pose_in.pose.orientation = computeQuaternionFromSurfaceNormal(handle["surface_normal"]);

        geometry_msgs::msg::PoseStamped pose_out;
        if (transformPose(pose_in, target_frame, pose_out, *tf_buffer, node->get_logger())) {
            pose_pub->publish(pose_out);
            RCLCPP_INFO(node->get_logger(),
                        "Published initial pose for handle %d (%s -> %s)",
                        handle_id, frame_id.c_str(), target_frame.c_str());
        }

        // -- Waypoints --
        for (const auto &wp : handle["waypoints"])
        {
            if (g_shutdown_requested) break;

            double delay_time = wp["t"].get<double>() - last_time;
            last_time = wp["t"].get<double>();
            rclcpp::sleep_for(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::duration<double>(delay_time)));

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
            if (transformPose(wp_in, target_frame, wp_out, *tf_buffer, node->get_logger())) {
                pose_pub->publish(wp_out);
                RCLCPP_INFO(node->get_logger(),
                            "Published waypoint t=%.2f for handle %d",
                            wp["t"].get<double>(), handle_id);
            }
        }
    }

    RCLCPP_INFO(node->get_logger(), "All handles processed.");
    rclcpp::shutdown();
    return 0;
}