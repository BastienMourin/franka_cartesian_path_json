#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <std_msgs/msg/string.hpp>                      // ← NEW: for run_folder topic
#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>                                    // ← NEW: for path manipulation
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <Eigen/Dense>
#include <atomic>
#include <signal.h>
#include <thread>
#include <map>
#include <chrono>                                        // ← NEW: for timestamp
#include <ctime>                                         // ← NEW: for localtime_r

// For subprocess management
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>

using json = nlohmann::json;
namespace fs = std::filesystem;                          // ← NEW

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


// ===========================================================================
// Run-folder creation
//
// Derives the run folder from the waypoints file path:
//   waypoints = /experiments/test1/action_profiles_displacement.json
//   records   = /experiments/test1/records
//   run_dir   = /experiments/test1/records/20260227_160837/
//       wrench/
//       obs/
//       realsense/
// ===========================================================================
std::string createRunFolder(const std::string &waypoints_abs, const rclcpp::Logger &logger)
{
    fs::path parent = fs::path(waypoints_abs).parent_path();
    fs::path records_root = parent / "records";

    auto now = std::chrono::system_clock::now();
    std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm_local{};
    localtime_r(&tt, &tm_local);
    char ts[32];
    std::strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", &tm_local);

    fs::path run_dir = records_root / ts;

    try {
        fs::create_directories(run_dir / "wrench");
        fs::create_directories(run_dir / "obs");
        fs::create_directories(run_dir / "realsense");
    } catch (const std::exception &e) {
        RCLCPP_ERROR(logger, "Failed to create run folder '%s': %s",
                     run_dir.string().c_str(), e.what());
        return "";
    }

    RCLCPP_INFO(logger, "Run folder created: %s", run_dir.string().c_str());
    RCLCPP_INFO(logger, "  Subfolders: wrench/  obs/  realsense/");
    return run_dir.string();
}


// ===========================================================================
// Wrench Logger subprocess management
// ===========================================================================

// ---------------------------------------------------------------------------
// Launch a ROS 2 node via `ros2 run` as a child process.
// The workspace must be sourced in the calling environment (standard when
// launching from a ROS 2 launch file or a sourced terminal).
//
// Optional ROS parameters can be forwarded via extra_ros_args, e.g.:
//   {"-p", "save_interval_sec:=10.0", "-p", "tf_timeout_sec:=0.1"}
//
// Returns the child PID on success, -1 on failure.
// ---------------------------------------------------------------------------
pid_t launchROSNode(
    const rclcpp::Logger              &logger,
    const std::string                 &package_name,
    const std::string                 &executable_name,
    const std::vector<std::string>    &extra_ros_args = {})
{
    pid_t pid = fork();

    if (pid < 0) {
        RCLCPP_ERROR(logger, "fork() failed when launching %s %s: %s",
                     package_name.c_str(), executable_name.c_str(), strerror(errno));
        return -1;
    }

    if (pid == 0)
    {
        // ---- CHILD PROCESS ------------------------------------------------
        // Build argv for:
        //   ros2 run <package> <executable> [--ros-args [-p key:=val ...]]
        std::vector<std::string> args_str = {
            "ros2", "run", package_name, executable_name
        };

        if (!extra_ros_args.empty()) {
            args_str.push_back("--ros-args");
            for (const auto &a : extra_ros_args)
                args_str.push_back(a);
        }

        // Convert to char* array required by execvp
        std::vector<const char *> argv_ptrs;
        argv_ptrs.reserve(args_str.size() + 1);
        for (const auto &s : args_str)
            argv_ptrs.push_back(s.c_str());
        argv_ptrs.push_back(nullptr);

        execvp("ros2", const_cast<char **>(argv_ptrs.data()));

        // execvp only returns on failure
        // Use write() instead of RCLCPP_ERROR — rclcpp is not available in child after fork
        const char *msg = "[launchROSNode child] execvp failed\n";
        write(STDERR_FILENO, msg, strlen(msg));
        _exit(1);
    }

    // ---- PARENT PROCESS ---------------------------------------------------
    RCLCPP_INFO(logger,
        "Launched %s %s via 'ros2 run' (child PID %d)",
        package_name.c_str(), executable_name.c_str(), pid);

    return pid;
}

// ---------------------------------------------------------------------------
// Gracefully stop a child process:
//   1. Send SIGINT  → triggers signal handler → allows for cleanup/flush
//   2. Poll for up to `timeout_ms` milliseconds
//   3. If still alive after timeout, send SIGKILL
// ---------------------------------------------------------------------------
void stopChildProcess(pid_t pid, const rclcpp::Logger &logger, const std::string &name = "", int timeout_ms = 3000)
{
    if (pid <= 0) return;

    std::string name_str = !name.empty() ? name : std::to_string(pid);
    RCLCPP_INFO(logger,
        "Sending SIGINT to %s (PID %d) — waiting for cleanup...", name_str.c_str(), pid);
    kill(pid, SIGINT);

    const int poll_interval_ms = 100;
    int elapsed_ms = 0;
    int status = 0;

    while (elapsed_ms < timeout_ms) {
        pid_t result = waitpid(pid, &status, WNOHANG);
        if (result == pid) {
            RCLCPP_INFO(logger,
                "%s (PID %d) exited cleanly (status %d).", name_str.c_str(), pid, status);
            return;
        }
        rclcpp::sleep_for(std::chrono::milliseconds(poll_interval_ms));
        elapsed_ms += poll_interval_ms;
    }

    // Still alive — force kill
    RCLCPP_WARN(logger,
        "%s (PID %d) did not exit within %d ms — sending SIGKILL.", name_str.c_str(), pid, timeout_ms);
    kill(pid, SIGKILL);
    waitpid(pid, &status, 0);
    RCLCPP_INFO(logger, "%s (PID %d) force-killed.", name_str.c_str(), pid);
}


// ===========================================================================
// Debug marker helpers
//
// Two topics per handle:
//   /debug/handle_<N>/waypoints_camera     – raw cumulative_position
//                                            expressed in camera frame
//                                            (LINE_STRIP + spheres)
//   /debug/handle_<N>/waypoints_link0      – computed target positions
//                                            in fr3_link0
//                                            (LINE_STRIP + spheres)
//
// Colours:
//   camera  waypoints → cyan   (0, 1, 1)
//   link0   waypoints → orange (1, 0.5, 0)
//
// Enable with ROS parameter:  --ros-args -p debug_markers:=true
// ===========================================================================

// Per-handle debug publisher pair
struct DebugPublishers {
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr camera_pub;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr link0_pub;
};

// Accumulate points; flushed once the whole trajectory is collected
struct DebugTrajectory {
    std::vector<geometry_msgs::msg::Point> camera_points;  // in camera frame
    std::vector<geometry_msgs::msg::Point> link0_points;   // in link0 frame
};

// Build a MarkerArray with a LINE_STRIP + individual SPHERE_LIST from a
// list of points.  id_offset separates the two channels so IDs don't clash.
visualization_msgs::msg::MarkerArray buildMarkerArray(
    const std::vector<geometry_msgs::msg::Point> &points,
    const std::string &frame_id,
    float r, float g, float b,
    int id_offset,
    const rclcpp::Time &stamp)
{
    visualization_msgs::msg::MarkerArray ma;

    if (points.empty()) return ma;

    // --- LINE_STRIP connecting all waypoints ---
    visualization_msgs::msg::Marker line;
    line.header.frame_id = frame_id;
    line.header.stamp    = stamp;
    line.ns              = "path";
    line.id              = id_offset;
    line.type            = visualization_msgs::msg::Marker::LINE_STRIP;
    line.action          = visualization_msgs::msg::Marker::ADD;
    line.scale.x         = 0.003;   // line width [m]
    line.color.r = r; line.color.g = g; line.color.b = b; line.color.a = 0.9f;
    line.pose.orientation.w = 1.0;
    line.points          = points;
    ma.markers.push_back(line);

    // --- SPHERE_LIST marking each waypoint ---
    visualization_msgs::msg::Marker spheres;
    spheres.header       = line.header;
    spheres.ns           = "waypoints";
    spheres.id           = id_offset + 1;
    spheres.type         = visualization_msgs::msg::Marker::SPHERE_LIST;
    spheres.action       = visualization_msgs::msg::Marker::ADD;
    spheres.scale.x = spheres.scale.y = spheres.scale.z = 0.008;
    spheres.color        = line.color;
    spheres.pose.orientation.w = 1.0;
    spheres.points       = points;
    ma.markers.push_back(spheres);

    // --- Larger SPHERE at first waypoint to show starting point ---
    if (!points.empty()) {
    visualization_msgs::msg::Marker start_sphere;
    start_sphere.header             = line.header;
    start_sphere.ns                 = "start";
    start_sphere.id                 = id_offset + 2;
    start_sphere.type               = visualization_msgs::msg::Marker::SPHERE;
    start_sphere.action             = visualization_msgs::msg::Marker::ADD;
    start_sphere.pose.position      = points.front();
    start_sphere.pose.orientation.w = 1.0;
    start_sphere.scale.x = start_sphere.scale.y = start_sphere.scale.z = 0.016;
    start_sphere.color.r = 1.0f; start_sphere.color.g = 1.0f;
        start_sphere.color.b = 0.0f; start_sphere.color.a = 1.0f;  // yellow = start
    ma.markers.push_back(start_sphere);
    }

    return ma;
}


// ===========================================================================
// Look up the current pose of source_frame expressed in
// target_frame via TF. Retries up to max_attempts times.
// ===========================================================================
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


// ===========================================================================
// Apply a delta pose on top of a base pose, both in fr3_link0.
//
// POSITION:
//   result.position = base.position + delta.position
//   The delta is added directly because the caller already rotated
//   it from the source frame into fr3_link0 (world frame).
//
// ORIENTATION:
//   result.orientation = q_delta * q_base   (pre-multiplication)
//   Pre-multiplying applies the delta rotation in the WORLD frame
//   (fr3_link0), meaning the rotation axis is fixed in space.
// ===========================================================================
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


// ===========================================================================
// Rotate a displacement vector from source_frame into target_frame,
// applying ROTATION ONLY (no translation/origin offset).
// ===========================================================================
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


// ===========================================================================
// Re-express a delta quaternion from source_frame into target_frame
// using the SIMILARITY TRANSFORM:
//   q_out = q_rot * q_delta * q_rot^{-1}
// ===========================================================================
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

    Eigen::Quaterniond q_out = (q_rot * q_delta * q_rot.inverse()).normalized();

    q_delta_out.w = q_out.w();
    q_delta_out.x = q_out.x();
    q_delta_out.y = q_out.y();
    q_delta_out.z = q_out.z();
    return true;
}


// ===========================================================================
// Per-handle publishing thread
// ===========================================================================
void publishHandle(
    rclcpp::Node::SharedPtr node,
    const json handle,
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub,
    std::shared_ptr<tf2_ros::Buffer> tf_buffer,
    const std::chrono::steady_clock::time_point t0,
    bool debug,
    DebugPublishers debug_pubs)
{
    int handle_id = handle["handle_id"].get<int>();

    std::string frame_id = "camera_color_optical_frame";  // default if not in JSON
    if (handle.contains("frame_id"))
        frame_id = handle["frame_id"].get<std::string>();

    std::string tcp_frame     = (handle_id == 0) ? "NS1_fr3_hand_tcp" : "NS2_fr3_hand_tcp";
    std::string control_frame = (handle_id == 0) ? "NS1_fr3_link0"    : "NS2_fr3_link0";

    geometry_msgs::msg::PoseStamped start_pose_in_link0;
    if (!getCurrentPose(tcp_frame, control_frame, start_pose_in_link0, *tf_buffer, node->get_logger()))
    {
        RCLCPP_ERROR(node->get_logger(),
            "Could not read current TCP pose in %s. Aborting handle %d.",
            control_frame.c_str(), handle_id);
        return;
    }

    RCLCPP_INFO(node->get_logger(),
        "[handle %d] Pre-computing trajectory markers...", handle_id);

    DebugTrajectory debug_traj;

    for (const auto &wp : handle["waypoints"])
    {
        if (g_shutdown_requested) break;

            // Camera-frame point (raw cumulative_position)
        geometry_msgs::msg::Point cam_pt;
        cam_pt.x = wp["cumulative_position"][0].get<double>();
        cam_pt.y = wp["cumulative_position"][1].get<double>();
        cam_pt.z = wp["cumulative_position"][2].get<double>();
        debug_traj.camera_points.push_back(cam_pt);

            // Link0-frame point (start_pose + rotated delta)
        geometry_msgs::msg::Vector3 raw_delta_pos;
        raw_delta_pos.x = wp["cumulative_position"][0].get<double>();
        raw_delta_pos.y = wp["cumulative_position"][1].get<double>();
        raw_delta_pos.z = wp["cumulative_position"][2].get<double>();

        geometry_msgs::msg::Point delta_in_link0;
        if (!rotateDeltaPosition(raw_delta_pos, frame_id, control_frame,
                                 delta_in_link0, *tf_buffer,
                                 node->get_clock(), node->get_logger()))
        {
            RCLCPP_WARN(node->get_logger(),
                "[handle %d] Could not rotate waypoint for marker, skipping.", handle_id);
            continue;
        }

        geometry_msgs::msg::Point link0_pt;
        link0_pt.x = start_pose_in_link0.pose.position.x + delta_in_link0.x;
        link0_pt.y = start_pose_in_link0.pose.position.y + delta_in_link0.y;
        link0_pt.z = start_pose_in_link0.pose.position.z + delta_in_link0.z;
        debug_traj.link0_points.push_back(link0_pt);
    }

        // Publish camera-frame markers (cyan)
    auto stamp = node->get_clock()->now();
    debug_pubs.camera_pub->publish(buildMarkerArray(
        debug_traj.camera_points, frame_id,
        0.0f, 1.0f, 1.0f, handle_id * 10, stamp));

    debug_pubs.link0_pub->publish(buildMarkerArray(
        debug_traj.link0_points, control_frame,
        1.0f, 0.5f, 0.0f, handle_id * 10, stamp));

    RCLCPP_INFO(node->get_logger(),
        "[handle %d] Markers published — %zu camera pts, %zu link0 pts.",
        handle_id,
        debug_traj.camera_points.size(),
        debug_traj.link0_points.size());

    if (debug) {
        RCLCPP_WARN(node->get_logger(),
            "[handle %d] debug=true — trajectory execution SKIPPED. "
            "Re-run with debug:=false to execute.", handle_id);
        return;
    }

    RCLCPP_INFO(node->get_logger(), "[handle %d] Executing trajectory...", handle_id);

    double last_t = 0.0;

    for (const auto &wp : handle["waypoints"])
    {
        if (g_shutdown_requested) break;

        // Timing
        double t         = wp["t_s"].get<double>();
        double delay_sec = t - last_t;
        last_t           = t;

        if (delay_sec > 0.0) {
            rclcpp::sleep_for(
                std::chrono::nanoseconds(static_cast<int64_t>(delay_sec * 1e9)));
        }

        // Read raw delta from JSON
        geometry_msgs::msg::Vector3 raw_delta_pos;
        raw_delta_pos.x = wp["cumulative_position"][0].get<double>();
        raw_delta_pos.y = wp["cumulative_position"][1].get<double>();
        raw_delta_pos.z = wp["cumulative_position"][2].get<double>();

        geometry_msgs::msg::Quaternion raw_delta_ori;
        raw_delta_ori.w = wp["cumulative_quaternion_wxyz"][0].get<double>();
        raw_delta_ori.x = wp["cumulative_quaternion_wxyz"][1].get<double>();
        raw_delta_ori.y = wp["cumulative_quaternion_wxyz"][2].get<double>();
        raw_delta_ori.z = wp["cumulative_quaternion_wxyz"][3].get<double>();

        // Rotate delta position into fr3_link0
        geometry_msgs::msg::Point delta_pos_in_link0;
        if (!rotateDeltaPosition(raw_delta_pos, frame_id, control_frame, delta_pos_in_link0,
                                 *tf_buffer, node->get_clock(), node->get_logger()))
        {
            RCLCPP_ERROR(node->get_logger(),
                "Aborting: could not rotate delta position at t=%.2f", t);
            return;
        }

        // Re-express delta orientation into fr3_link0
        geometry_msgs::msg::Quaternion delta_ori_in_link0;
        if (!rotateDeltaOrientation(raw_delta_ori, frame_id, control_frame, delta_ori_in_link0,
                                    *tf_buffer, node->get_logger()))
        {
            RCLCPP_ERROR(node->get_logger(),
                "Aborting: could not rotate delta orientation at t=%.2f", t);
            return;
        }

        // Apply delta on top of starting TCP pose
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
        RCLCPP_INFO(node->get_logger(), "[handle %d] Published target pose at t=%.2f", handle_id, t);
    }

    RCLCPP_INFO(node->get_logger(), "[handle %d] Done.", handle_id);
}


// ===========================================================================
// MAIN
// ===========================================================================
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = rclcpp::Node::make_shared("franka_cartesian_path_json_dual_increments");
    signal(SIGINT, signalHandler);

    // ------------------------------------------------------------------
    // Parameters
    // ------------------------------------------------------------------
    node->declare_parameter("debug", false);
    bool debug = node->get_parameter("debug").as_bool();

    if (debug)
        RCLCPP_WARN(node->get_logger(),
            "debug=true — markers will be published but the robot will NOT move. "
            "Subscribe to /debug/handle_<N>/waypoints_camera and "
            "/debug/handle_<N>/waypoints_link0 in RViz.");
    else
        RCLCPP_INFO(node->get_logger(),
            "debug=false — markers will be published and trajectory will execute. "
            "Subscribe to /debug/handle_<N>/waypoints_camera and "
            "/debug/handle_<N>/waypoints_link0 in RViz.");

    node->declare_parameter("enable_recording", false);
    bool enable_recording = node->get_parameter("enable_recording").as_bool();

    // ------------------------------------------------------------------
    // NEW: Resolve waypoints file to an absolute path
    //
    // Accepts either:
    //   absolute path  →  used directly   e.g. /home/.../experiments/test1/action_profiles.json
    //   relative path  →  resolved relative to package share directory
    // ------------------------------------------------------------------
    node->declare_parameter("waypoints_file", "action_profiles_displacement.json");
    std::string waypoints_param = node->get_parameter("waypoints_file").as_string();

    std::string waypoints_abs;
    if (!waypoints_param.empty() && waypoints_param[0] == '/') {
        waypoints_abs = waypoints_param;
    } else {
        std::string pkg = ament_index_cpp::get_package_share_directory(
            "ignacio_cartesian_pose_json");
        waypoints_abs = pkg + "/" + waypoints_param;
    }
    RCLCPP_INFO(node->get_logger(), "Loading waypoints from: %s", waypoints_abs.c_str());

    // ------------------------------------------------------------------
    // Launch recording pipeline: obs_recorder, realsense_recorder, and manager
    // Only create timestamped folder if recording is enabled
    // ------------------------------------------------------------------
    std::vector<pid_t> recorder_pids;  // obs + realsense
    pid_t manager_pid = -1;
    if (enable_recording && !debug) {
        RCLCPP_INFO(node->get_logger(), "Starting recording pipeline...");

        // Create timestamped run folder next to the waypoints file
        // and publish it so Python recorder nodes know where to save
        std::string run_folder = createRunFolder(waypoints_abs, node->get_logger());
        if (run_folder.empty()) {
            rclcpp::shutdown();
            return 1;
        }

        // Transient-local (latched) — nodes that start after this still receive it
        auto run_folder_pub = node->create_publisher<std_msgs::msg::String>(
            "/recording_manager/run_folder",
            rclcpp::QoS(1).transient_local());

        std_msgs::msg::String rf_msg;
        rf_msg.data = run_folder;
        run_folder_pub->publish(rf_msg);
        RCLCPP_INFO(node->get_logger(),
            "Published run folder on /recording_manager/run_folder: %s", run_folder.c_str());

        // Brief pause so Python nodes can receive the latched message
        rclcpp::sleep_for(std::chrono::milliseconds(500));

        // Launch obs_recorder_node with its save dir pre-set and auto_record on
        pid_t obs_pid = launchROSNode(
            node->get_logger(),
            "ignacio_cartesian_pose_json",
            "obs_recorder_node",
            {"-p", std::string("data_dir:=") + run_folder + "/obs",
             "-p", "auto_record:=true"});
        if (obs_pid > 0) recorder_pids.push_back(obs_pid);

        rclcpp::sleep_for(std::chrono::milliseconds(500));

        // Launch realsense_recorder_node with its save dir pre-set and auto_record on
        pid_t realsense_pid = launchROSNode(
            node->get_logger(),
            "ignacio_cartesian_pose_json",
            "realsense_recorder_node",
            {"-p", std::string("data_dir:=") + run_folder + "/realsense",
             "-p", "auto_record:=true"});
        if (realsense_pid > 0) recorder_pids.push_back(realsense_pid);

        rclcpp::sleep_for(std::chrono::milliseconds(500));

        // Launch recording_manager_node with auto_record enabled.
        // Stopped FIRST during shutdown so it can gracefully stop obs/realsense
        // via services, copy the OBS file, then exit before we kill the others.
        manager_pid = launchROSNode(
            node->get_logger(),
            "ignacio_cartesian_pose_json",
            "recording_manager_node",
            {"-p", "auto_record:=true"});
        // manager_pid intentionally not pushed into recorder_pids here;
        // it is stored separately and stopped before the rest.

        RCLCPP_INFO(node->get_logger(),
            "Recording pipeline started (%zu processes). Waiting for initialization...",
            recorder_pids.size());
        rclcpp::sleep_for(std::chrono::seconds(2));
    } else if (enable_recording && debug) {
        RCLCPP_WARN(node->get_logger(), "Recording pipeline disabled because debug=true.");
    } else {
        RCLCPP_INFO(node->get_logger(), "Recording pipeline disabled.");
    }

    // ------------------------------------------------------------------
    // TF2 buffer + listener
    // ------------------------------------------------------------------
    auto tf_buffer   = std::make_shared<tf2_ros::Buffer>(node->get_clock());
    auto tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);

    RCLCPP_INFO(node->get_logger(), "Waiting 2 s for TF to become available...");
    rclcpp::sleep_for(std::chrono::seconds(2));

    // ------------------------------------------------------------------
    // Create one pose publisher + one debug publisher pair per handle
    // ------------------------------------------------------------------
    std::map<int, rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr> publishers;
    std::map<int, DebugPublishers> debug_pub_map;

    for (const auto &[handle_id, cfg] : HANDLE_CONFIG) {
        std::string topic = cfg.ns + "/my_cartesian_impedance_controller/target_pose";
        publishers[handle_id] =
            node->create_publisher<geometry_msgs::msg::PoseStamped>(topic, 10);
        RCLCPP_INFO(node->get_logger(),
            "Publisher handle_id %d -> %s  (base frame: %s)",
            handle_id, topic.c_str(), cfg.base_frame.c_str());

        std::string cam_topic   = "/debug/handle_" + std::to_string(handle_id) + "/waypoints_camera";
        std::string link0_topic = "/debug/handle_" + std::to_string(handle_id) + "/waypoints_link0";
            // transient_local = latched: late subscribers still receive the message
            // depth=10 ensures RViz can retrieve markers even after toggling display
        auto qos = rclcpp::QoS(10).transient_local();
        debug_pub_map[handle_id] = {
            node->create_publisher<visualization_msgs::msg::MarkerArray>(cam_topic,   qos),
            node->create_publisher<visualization_msgs::msg::MarkerArray>(link0_topic, qos)
        };
    }

    // ------------------------------------------------------------------
    // Load JSON waypoints file
    // ------------------------------------------------------------------
    std::ifstream file(waypoints_abs);
    if (!file.is_open()) {
        RCLCPP_ERROR(node->get_logger(), "Cannot open JSON file: %s", waypoints_abs.c_str());
        if (manager_pid > 0) stopChildProcess(manager_pid, node->get_logger(), "recording_manager", 20000);
        for (pid_t pid : recorder_pids) stopChildProcess(pid, node->get_logger());
        return 1;
    }
    json j;
    file >> j;

    // ------------------------------------------------------------------
    // Spawn one thread per handle and execute the trajectory
    // ------------------------------------------------------------------
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> threads;

    for (const auto &handle : j["handles"])
    {
        int handle_id = handle["handle_id"].get<int>();

        auto cfg_it = HANDLE_CONFIG.find(handle_id);
        auto pub_it = publishers.find(handle_id);
        auto dbg_it = debug_pub_map.find(handle_id);

        if (cfg_it == HANDLE_CONFIG.end() || pub_it == publishers.end()) {
            RCLCPP_WARN(node->get_logger(),
                "No config/publisher for handle_id %d — skipping.", handle_id);
            continue;
        }

        DebugPublishers dbg_pubs = (dbg_it != debug_pub_map.end()) ? dbg_it->second : DebugPublishers{};

        threads.emplace_back(publishHandle,
                             node,
                             handle,
                             pub_it->second,
                             tf_buffer,
                             t0,
                             debug,
                             dbg_pubs);
    }

    for (auto &t : threads) t.join();

    if (debug)
        RCLCPP_INFO(node->get_logger(),
            "All markers published. Inspect in RViz, then re-run with debug:=false to execute.");
    else
        RCLCPP_INFO(node->get_logger(), "All handles processed.");

    // ------------------------------------------------------------------
    // Trajectory is complete — stop all recorder processes gracefully.
    // Stop manager FIRST: it calls stop services on obs/realsense and copies
    // the OBS file. Then stop obs and realsense (likely already stopped by
    // the manager, but kill any survivors).
    // ------------------------------------------------------------------
    if (manager_pid > 0 || !recorder_pids.empty()) {
        RCLCPP_INFO(node->get_logger(), "Stopping recorder processes...");

        // 1. Manager — handles graceful stop + OBS file copy
        if (manager_pid > 0)
            stopChildProcess(manager_pid, node->get_logger(), "recording_manager", 20000);

        // 2. obs + realsense — kill survivors after manager is done
        for (pid_t pid : recorder_pids)
            stopChildProcess(pid, node->get_logger(), "", 3000);
    }

    rclcpp::shutdown();
    return 0;
}