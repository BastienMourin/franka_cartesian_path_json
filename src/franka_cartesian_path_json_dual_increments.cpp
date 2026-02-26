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
#include <thread>
#include <map>

using json = nlohmann::json;

std::atomic<bool> g_shutdown_requested(false);
void signalHandler(int) { g_shutdown_requested = true; }

// Map handle_id -> { topic namespace, TF target frame }
struct HandleConfig {
    std::string ns;           // e.g. "/NS1"
    std::string base_frame;   // e.g. "NS1_fr3_link0"
};

static const std::map<int, HandleConfig> HANDLE_CONFIG = {
    {0, {"/NS1", "NS1_fr3_link0"}},
    {1, {"/NS2", "NS2_fr3_link0"}},
};

// ==========================================================
// Look up the current pose of source_frame expressed in
// target_frame via TF. Retries up to max_attempts times.
// ==========================================================
bool getCurrentPose(
    const std::string               &source_frame,
    const std::string               &target_frame,
    geometry_msgs::msg::PoseStamped &pose_out,
    tf2_ros::Buffer                 &tf_buffer,
    const rclcpp::Logger            &logger,
    int max_attempts = 10)
{
    for (int attempt = 0; attempt < max_attempts; ++attempt)
    {
        try {
            auto tf = tf_buffer.lookupTransform(
                target_frame, source_frame,
                tf2::TimePointZero,
                tf2::durationFromSec(0.5));

            pose_out.header.frame_id  = target_frame;
            pose_out.header.stamp     = tf.header.stamp;
            pose_out.pose.position.x  = tf.transform.translation.x;
            pose_out.pose.position.y  = tf.transform.translation.y;
            pose_out.pose.position.z  = tf.transform.translation.z;
            pose_out.pose.orientation = tf.transform.rotation;

            RCLCPP_INFO(logger,
                "Current pose of '%s' in '%s': pos=[%.3f, %.3f, %.3f] quat=[%.3f, %.3f, %.3f, %.3f]",
                source_frame.c_str(), target_frame.c_str(),
                pose_out.pose.position.x,
                pose_out.pose.position.y,
                pose_out.pose.position.z,
                pose_out.pose.orientation.x,
                pose_out.pose.orientation.y,
                pose_out.pose.orientation.z,
                pose_out.pose.orientation.w);

            return true;

        } catch (const tf2::TransformException &ex) {
            RCLCPP_WARN(logger,
                "Attempt %d/%d – TF '%s' -> '%s' failed: %s",
                attempt + 1, max_attempts,
                source_frame.c_str(), target_frame.c_str(), ex.what());
            rclcpp::sleep_for(std::chrono::milliseconds(200));
        }
    }
    return false;
}

// ==========================================================
// Apply a delta pose on top of a base pose, both in fr3_link0.
//
// POSITION:
//   result.position = base.position + delta.position
//   The delta is added directly because the caller already rotated
//   it from the source frame into fr3_link0 (world frame).
//   No additional rotation by q_base is needed here.
//
// ORIENTATION:
//   result.orientation = q_delta * q_base   (pre-multiplication)
//   Pre-multiplying applies the delta rotation in the WORLD frame
//   (fr3_link0), meaning the rotation axis is fixed in space.
//   This is correct because the caller already re-expressed the
//   delta axis from the source frame into fr3_link0 via the
//   similarity transform (see rotateDeltaOrientation).
//
//   If you post-multiply (q_base * q_delta) instead, the rotation
//   is applied in the END-EFFECTOR LOCAL frame, which would ignore
//   the frame_id entirely and always rotate around the EEF axes.
// ==========================================================
geometry_msgs::msg::Pose applyDeltaPose(
    const geometry_msgs::msg::Pose &base,
    const geometry_msgs::msg::Pose &delta,
    const rclcpp::Logger           &logger)
{
    Eigen::Vector3d new_position(
        base.position.x + delta.position.x,
        base.position.y + delta.position.y,
        base.position.z + delta.position.z);

    Eigen::Quaterniond q_base(
        base.orientation.w,
        base.orientation.x,
        base.orientation.y,
        base.orientation.z);
    Eigen::Quaterniond q_delta(
        delta.orientation.w,
        delta.orientation.x,
        delta.orientation.y,
        delta.orientation.z);
    Eigen::Quaterniond q_result = (q_delta * q_base).normalized();

    geometry_msgs::msg::Pose result;
    result.position.x    = new_position.x();
    result.position.y    = new_position.y();
    result.position.z    = new_position.z();
    result.orientation.w = q_result.w();
    result.orientation.x = q_result.x();
    result.orientation.y = q_result.y();
    result.orientation.z = q_result.z();

    RCLCPP_INFO(logger,
        "Updated pose in fr3_link0: pos=[%.4f, %.4f, %.4f] quat=[%.4f, %.4f, %.4f, %.4f]",
        result.position.x, result.position.y, result.position.z,
        result.orientation.x, result.orientation.y, result.orientation.z, result.orientation.w);

    return result;
}

// ==========================================================
// Rotate a displacement vector from source_frame into target_frame,
// applying ROTATION ONLY (no translation/origin offset).
//
// Uses Vector3Stamped which TF transforms by rotating the vector
// direction without adding the translation between frame origins.
// This is the correct way to handle a delta/displacement: a vector
// [0, 0, 0.02] in camera-Z becomes the equivalent direction in
// fr3_link0, without any spurious offset from where the camera sits.
//
// Compare with transforming a PoseStamped (WRONG for deltas): that
// would also add the full camera→link0 translation, turning a tiny
// delta into a large jump on the first waypoint.
// ==========================================================
bool rotateDeltaPosition(
    const geometry_msgs::msg::Vector3 &delta_in,
    const std::string                 &source_frame,
    const std::string                 &target_frame,
    geometry_msgs::msg::Point         &delta_out,
    tf2_ros::Buffer                   &tf_buffer,
    const rclcpp::Clock::SharedPtr    &clock,
    const rclcpp::Logger              &logger)
{
    if (source_frame == target_frame) {
        delta_out.x = delta_in.x;
        delta_out.y = delta_in.y;
        delta_out.z = delta_in.z;
        return true;
    }

    geometry_msgs::msg::Vector3Stamped v_in, v_out;
    v_in.header.frame_id = source_frame;
    v_in.header.stamp    = clock->now();
    v_in.vector          = delta_in;

    try {
        v_out = tf_buffer.transform(v_in, target_frame, tf2::durationFromSec(0.1));
        delta_out.x = v_out.vector.x;
        delta_out.y = v_out.vector.y;
        delta_out.z = v_out.vector.z;
        return true;
    } catch (const tf2::TransformException &ex) {
        RCLCPP_WARN(logger,
            "Failed to rotate delta position from '%s' to '%s': %s",
            source_frame.c_str(), target_frame.c_str(), ex.what());
        return false;
    }
}

// ==========================================================
// Re-express a delta quaternion from source_frame into target_frame
// using the SIMILARITY TRANSFORM:
//
//   q_out = q_rot * q_delta * q_rot^{-1}
//
// where q_rot is the rotation from source_frame to target_frame.
//
// WHY NOT use tf_buffer.transform() on a QuaternionStamped?
//   That function interprets the quaternion as an absolute orientation
//   and applies the full rigid body transform, adding the camera's
//   own mounting orientation as an offset. A JSON delta of [1,0,0,0]
//   (identity = no rotation) would become the camera's orientation in
//   link0 — causing unexpected rotation even when none is requested.
//
// WHY NOT pass raw_delta_ori through unchanged?
//   That always interprets the rotation axis in fr3_link0 coordinates,
//   ignoring frame_id entirely. "Rotate around camera-Z" would behave
//   identically to "rotate around link0-Z".
//
// THE SIMILARITY TRANSFORM correctly re-expresses only the rotation
//   AXIS into the new frame, leaving the rotation ANGLE unchanged.
//   "Rotate 45° around camera-Z" becomes "rotate 45° around the
//   camera-Z axis as expressed in fr3_link0 coordinates".
//   Identity [1,0,0,0] remains identity after the transform.
// ==========================================================
bool rotateDeltaOrientation(
    const geometry_msgs::msg::Quaternion &q_delta_in,
    const std::string                    &source_frame,
    const std::string                    &target_frame,
    geometry_msgs::msg::Quaternion       &q_delta_out,
    tf2_ros::Buffer                      &tf_buffer,
    const rclcpp::Logger                 &logger)
{
    if (source_frame == target_frame) {
        q_delta_out = q_delta_in;
        return true;
    }

    // Look up the rotation from source_frame to target_frame
    geometry_msgs::msg::TransformStamped tf_stamped;
    try {
        tf_stamped = tf_buffer.lookupTransform(
            target_frame, source_frame,
            tf2::TimePointZero,
            tf2::durationFromSec(0.1));
    } catch (const tf2::TransformException &ex) {
        RCLCPP_WARN(logger,
            "Failed to look up rotation from '%s' to '%s': %s",
            source_frame.c_str(), target_frame.c_str(), ex.what());
        return false;
    }

    Eigen::Quaterniond q_rot(
        tf_stamped.transform.rotation.w,
        tf_stamped.transform.rotation.x,
        tf_stamped.transform.rotation.y,
        tf_stamped.transform.rotation.z);

    Eigen::Quaterniond q_delta(
        q_delta_in.w,
        q_delta_in.x,
        q_delta_in.y,
        q_delta_in.z);

    // Similarity transform: rotate the axis into target_frame
    Eigen::Quaterniond q_out = (q_rot * q_delta * q_rot.inverse()).normalized();

    q_delta_out.w = q_out.w();
    q_delta_out.x = q_out.x();
    q_delta_out.y = q_out.y();
    q_delta_out.z = q_out.z();
    return true;
}


// ==========================================================
// Per-handle publishing thread
// ==========================================================
void publishHandle(
    rclcpp::Node::SharedPtr node,
    const json handle,
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub,
    std::shared_ptr<tf2_ros::Buffer> tf_buffer,
    const std::string target_frame,               // e.g. "NS1_fr3_link0"
    const std::chrono::steady_clock::time_point t0)
{
    int handle_id = handle["handle_id"].get<int>();

    std::string frame_id = target_frame;
    if (handle.contains("frame_id"))
        frame_id = handle["frame_id"].get<std::string>();

    // ----------------------------------------------------------
    // Capture the starting TCP pose in the specific control frame
    // ----------------------------------------------------------
    geometry_msgs::msg::PoseStamped start_pose_in_link0;
    std::string tcp_frame = (handle_id == 0) ? "NS1_fr3_hand_tcp" : "NS2_fr3_hand_tcp";
    std::string control_frame = (handle_id == 0) ? "NS1_fr3_link0" : "NS2_fr3_link0";

    if (!getCurrentPose(tcp_frame, control_frame, start_pose_in_link0, *tf_buffer, node->get_logger()))
    {
        RCLCPP_ERROR(node->get_logger(),
            "Could not read current TCP pose in %s. Aborting.", control_frame.c_str());
        return;
    }

    double last_t = 0.0;

    // Process waypoints
    for (const auto &wp : handle["waypoints"])
    {
        if (g_shutdown_requested) break;

        // --- Timing ---
        double t         = wp["t"].get<double>();
        double delay_sec = t - last_t;
        last_t           = t;

        if (delay_sec > 0.0) {
            rclcpp::sleep_for(
                std::chrono::nanoseconds(static_cast<int64_t>(delay_sec * 1e9)));
        }

        // Read raw delta from JSON (expressed in frame_id)
        geometry_msgs::msg::Vector3 raw_delta_pos;
        raw_delta_pos.x = wp["position"][0].get<double>();
        raw_delta_pos.y = wp["position"][1].get<double>();
        raw_delta_pos.z = wp["position"][2].get<double>();

        geometry_msgs::msg::Quaternion raw_delta_ori;
        raw_delta_ori.w = wp["quaternion_wxyz"][0].get<double>();
        raw_delta_ori.x = wp["quaternion_wxyz"][1].get<double>();
        raw_delta_ori.y = wp["quaternion_wxyz"][2].get<double>();
        raw_delta_ori.z = wp["quaternion_wxyz"][3].get<double>();

        // Rotate delta position into fr3_link0
        geometry_msgs::msg::Point delta_pos_in_link0;
        if (!rotateDeltaPosition(raw_delta_pos, frame_id, control_frame, delta_pos_in_link0, *tf_buffer,
                                 node->get_clock(), node->get_logger()))
        {
            RCLCPP_ERROR(node->get_logger(),
                "Aborting: could not rotate delta position at t=%.2f", t);
            return;
        }

        // Re-express delta orientation into fr3_link0
        geometry_msgs::msg::Quaternion delta_ori_in_link0;
        if (!rotateDeltaOrientation(raw_delta_ori, frame_id, control_frame, delta_ori_in_link0, *tf_buffer,
                                    node->get_logger()))
        {
            RCLCPP_ERROR(node->get_logger(),
                "Aborting: could not rotate delta orientation at t=%.2f", t);
            return;
        }

        // Apply delta pose on top of the starting TCP pose
        geometry_msgs::msg::Pose delta_in_link0;
        delta_in_link0.position    = delta_pos_in_link0;
        delta_in_link0.orientation = delta_ori_in_link0;

        geometry_msgs::msg::Pose target_pose =
            applyDeltaPose(start_pose_in_link0.pose, delta_in_link0, node->get_logger());

        // Publish
        geometry_msgs::msg::PoseStamped target_stamped;
        target_stamped.header.frame_id = control_frame;
        target_stamped.header.stamp    = node->get_clock()->now();
        target_stamped.pose            = target_pose;

        pub->publish(target_stamped);
        RCLCPP_INFO(node->get_logger(), "Published target pose at t=%.2f", t);
    }

    RCLCPP_INFO(node->get_logger(), "[handle %d] Done.", handle_id);
}

// ==========================================================
// MAIN
// ==========================================================
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = rclcpp::Node::make_shared("franka_cartesian_path_json_dual_increments");
    signal(SIGINT, signalHandler);

    // TF2 buffer + listener
    auto tf_buffer   = std::make_shared<tf2_ros::Buffer>(node->get_clock());
    auto tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);

    RCLCPP_INFO(node->get_logger(), "Waiting 2 s for TF to become available...");
    rclcpp::sleep_for(std::chrono::seconds(2));

    // Create one publisher per handle_id / namespace
    std::map<int, rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr> publishers;

    for (const auto &[handle_id, cfg] : HANDLE_CONFIG) {
        std::string topic = cfg.ns + "/my_cartesian_impedance_controller/target_pose";
        publishers[handle_id] =
            node->create_publisher<geometry_msgs::msg::PoseStamped>(topic, 10);
        RCLCPP_INFO(node->get_logger(),
                    "Publisher handle_id %d -> %s  (base frame: %s)",
                    handle_id, topic.c_str(), cfg.base_frame.c_str());
    }

    // Load JSON
    std::string package_path =
        ament_index_cpp::get_package_share_directory("ignacio_cartesian_pose_json");

    node->declare_parameter("waypoints_file", "waypoints_test_dual_increments_rotation.json");
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

    // Spawn one thread per handle
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> threads;

    for (const auto &handle : j["handles"])
    {
        int handle_id = handle["handle_id"].get<int>();

        auto cfg_it = HANDLE_CONFIG.find(handle_id);
        auto pub_it = publishers.find(handle_id);

        if (cfg_it == HANDLE_CONFIG.end() || pub_it == publishers.end()) {
            RCLCPP_WARN(node->get_logger(),
                        "No config/publisher for handle_id %d — skipping.", handle_id);
            continue;
        }

        threads.emplace_back(publishHandle,
                             node,
                             handle,                      // json copied per thread
                             pub_it->second,
                             tf_buffer,                   // shared_ptr: thread-safe
                             cfg_it->second.base_frame,   // "NS1_fr3_link0" or "NS2_fr3_link0"
                             t0);
    }

    for (auto &t : threads) t.join();

    RCLCPP_INFO(node->get_logger(), "All handles processed.");
    rclcpp::shutdown();
    return 0;
}
