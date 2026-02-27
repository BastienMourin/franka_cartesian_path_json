#!/usr/bin/env python3
"""
realsense_recorder_node.py — saves RGB + aligned depth as MP4 videos
=====================================================================

Save path priority:
  1. ROS param  data_dir   — absolute path used directly
  2. Topic      /recording_manager/run_folder  — appends /realsense/
  3. Fallback   default root / timestamped folder

Saves two MP4 files:
  <save_dir>/color.mp4    — raw RGB (BGR, 8-bit)
  <save_dir>/depth.mp4    — depth visualised as greyscale

Services:
  /realsense_recorder/start   (std_srvs/srv/Trigger)
  /realsense_recorder/stop    (std_srvs/srv/Trigger)

Status:
  /realsense_recorder/status  (std_msgs/msg/String)

Parameters:
  data_dir        str     ""
  auto_record     bool    false
  fps             float   30.0
  rgb_topic       str     /camera/camera/color/image_raw
  depth_topic     str     /camera/camera/aligned_depth_to_color/image_raw
  status_rate_hz  float   1.0

Dependencies:
  sudo apt install ros-humble-cv-bridge python3-opencv
"""

import os
import signal
import sys
import threading
from datetime import datetime

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, ReliabilityPolicy, HistoryPolicy,
    DurabilityPolicy,
)
from std_msgs.msg import String
from std_srvs.srv import Trigger
from sensor_msgs.msg import Image

try:
    from cv_bridge import CvBridge
except ImportError:
    print('[realsense_recorder_node] ERROR: sudo apt install ros-humble-cv-bridge')
    sys.exit(1)

_DEFAULT_RGB   = '/camera/camera/color/image_raw'
_DEFAULT_DEPTH = '/camera/camera/aligned_depth_to_color/image_raw'


class RealSenseRecorderNode(Node):

    def __init__(self):
        super().__init__('realsense_recorder_node')
        self._shutdown_called = False
        self._lock            = threading.Lock()
        self._bridge          = CvBridge()

        self._recording        = False
        self._save_dir: str | None = None
        self._writer_color: cv2.VideoWriter | None = None
        self._writer_depth: cv2.VideoWriter | None = None
        self._color_frames = 0
        self._depth_frames = 0

        # ── Parameters ───────────────────────────────────────────────────────
        self.declare_parameter('data_dir',       '')
        self.declare_parameter('auto_record',    False)
        self.declare_parameter('fps',            30.0)
        self.declare_parameter('rgb_topic',      _DEFAULT_RGB)
        self.declare_parameter('depth_topic',    _DEFAULT_DEPTH)
        self.declare_parameter('status_rate_hz', 1.0)

        data_dir        = self.get_parameter('data_dir').get_parameter_value().string_value
        auto_record     = self.get_parameter('auto_record').get_parameter_value().bool_value
        self._fps       = self.get_parameter('fps').get_parameter_value().double_value
        rgb_topic       = self.get_parameter('rgb_topic').get_parameter_value().string_value
        depth_topic     = self.get_parameter('depth_topic').get_parameter_value().string_value
        rate_hz         = self.get_parameter('status_rate_hz').get_parameter_value().double_value
        self._fourcc    = cv2.VideoWriter_fourcc(*'mp4v')

        # ── Resolve save dir ─────────────────────────────────────────────────
        # Priority: data_dir > topic > fallback
        if data_dir:
            self._set_save_dir(data_dir)
        else:
            latched = QoSProfile(
                depth=1, history=HistoryPolicy.KEEP_LAST,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                reliability=ReliabilityPolicy.RELIABLE)
            self._folder_sub = self.create_subscription(
                String, '/recording_manager/run_folder',
                self._on_run_folder, latched)
            self.get_logger().info('Waiting for /recording_manager/run_folder ...')
            self._fallback_timer = self.create_timer(10.0, self._fallback_init)

        # ── Camera subscribers (best-effort matches RealSense QoS) ───────────
        cam_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Image, rgb_topic,   self._color_cb,  cam_qos)
        self.create_subscription(Image, depth_topic, self._depth_cb,  cam_qos)

        # ── Publisher / services ─────────────────────────────────────────────
        self._status_pub = self.create_publisher(String, 'realsense_recorder/status', 10)
        self.create_timer(1.0 / rate_hz, self._publish_status)
        self.create_service(Trigger, 'realsense_recorder/start', self._handle_start)
        self.create_service(Trigger, 'realsense_recorder/stop',  self._handle_stop)

        # ── Diagnostic timer (log frame counts every 5 sec while recording) ──
        self.create_timer(5.0, self._log_frame_counts)

        signal.signal(signal.SIGINT,  self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.get_logger().info('realsense_recorder_node ready.')
        self.get_logger().info(f'  RGB   : {rgb_topic}')
        self.get_logger().info(f'  Depth : {depth_topic}')
        self.get_logger().info(f'  FPS   : {self._fps}')

        if auto_record:
            self.get_logger().info('auto_record=true — starting recording now.')
            ok, msg = self._start_recording()
            if not ok:
                self.get_logger().error(f'auto_record failed: {msg}')

    # ── Directory helpers ─────────────────────────────────────────────────────

    def _set_save_dir(self, path: str):
        os.makedirs(path, exist_ok=True)
        self._save_dir = path
        self.get_logger().info(f'Save dir: {self._save_dir}')

    def _on_run_folder(self, msg: String):
        if self._save_dir is not None:
            return
        if hasattr(self, '_fallback_timer'):
            self._fallback_timer.cancel()
        self._set_save_dir(os.path.join(msg.data, 'realsense'))

    def _fallback_init(self):
        self._fallback_timer.cancel()
        if self._save_dir is not None:
            return
        # Fallback: use workspace-level records folder
        ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
        pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        fallback_root = os.path.join(pkg_root, 'records')
        path = os.path.join(fallback_root, ts, 'realsense')
        self.get_logger().warn(f'run_folder topic not received — fallback: {path}')
        self._set_save_dir(path)

    # ── Recording control ─────────────────────────────────────────────────────

    def _start_recording(self) -> tuple[bool, str]:
        with self._lock:
            if self._recording:
                return False, 'Already recording.'
            if self._save_dir is None:
                return False, 'Save dir not yet known (waiting for run_folder).'
            
            # Ensure save directory exists
            os.makedirs(self._save_dir, exist_ok=True)
            
            self._color_frames = 0
            self._depth_frames = 0
            self._writer_color = None   # created lazily on first frame
            self._writer_depth = None
            self._recording    = True
        self.get_logger().info(f'RealSense recording STARTED → {self._save_dir}')
        self.get_logger().info(f'  Waiting for images on {self.get_parameter("rgb_topic").get_parameter_value().string_value}')
        self.get_logger().info(f'                 and {self.get_parameter("depth_topic").get_parameter_value().string_value}')
        return True, f'Recording started → {self._save_dir}'

    def _stop_recording(self) -> tuple[bool, str]:
        with self._lock:
            if not self._recording:
                return False, 'Not recording.'
            self._recording = False
            if self._writer_color:
                self._writer_color.release()
                self._writer_color = None
            if self._writer_depth:
                self._writer_depth.release()
                self._writer_depth = None
            n_col, n_dep, d = self._color_frames, self._depth_frames, self._save_dir
        msg = f'Stopped — {n_col} color, {n_dep} depth frames → {d}'
        self.get_logger().info(msg)
        return True, msg

    # ── Image callbacks ───────────────────────────────────────────────────────

    def _color_cb(self, msg: Image):
        with self._lock:
            if not self._recording:
                return
            try:
                frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            except Exception as e:
                self.get_logger().warn(f'Color convert: {e}', throttle_duration_sec=5.0)
                return
            if self._writer_color is None:
                h, w = frame.shape[:2]
                path = os.path.join(self._save_dir, 'color.mp4')
                self._writer_color = cv2.VideoWriter(path, self._fourcc, self._fps, (w, h))
                self.get_logger().info(f'Color writer: {path} ({w}x{h}@{self._fps}fps)')
            self._writer_color.write(frame)
            self._color_frames += 1

    def _depth_cb(self, msg: Image):
        with self._lock:
            if not self._recording:
                return
            try:
                if msg.encoding in ('16UC1', '16U'):
                    depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough').astype(np.uint16)
                else:
                    depth = (self._bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1') * 1000.0).astype(np.uint16)
            except Exception:
                try:
                    depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                except Exception as e:
                    self.get_logger().warn(f'Depth convert: {e}', throttle_duration_sec=5.0)
                    return

            # Normalise to 8-bit greyscale for MP4
            d = depth.astype(np.float32)
            dmin, dmax = float(d.min()), float(d.max())
            disp = ((d - dmin) / (dmax - dmin) * 255.0).astype(np.uint8) \
                   if dmax > dmin else np.zeros_like(d, dtype=np.uint8)
            disp_bgr = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)

            if self._writer_depth is None:
                h, w = disp_bgr.shape[:2]
                path = os.path.join(self._save_dir, 'depth.mp4')
                self._writer_depth = cv2.VideoWriter(path, self._fourcc, self._fps, (w, h))
                self.get_logger().info(f'Depth writer: {path} ({w}x{h}@{self._fps}fps)')
            self._writer_depth.write(disp_bgr)
            self._depth_frames += 1

    # ── Service handlers / status ─────────────────────────────────────────────

    def _handle_start(self, request, response):
        response.success, response.message = self._start_recording()
        return response

    def _handle_stop(self, request, response):
        response.success, response.message = self._stop_recording()
        return response

    def _publish_status(self):
        msg = String()
        if self._recording and self._save_dir:
            msg.data = (f'recording | color={self._color_frames} '
                        f'depth={self._depth_frames} → {self._save_dir}')
        else:
            msg.data = 'idle'
        self._status_pub.publish(msg)

    def _log_frame_counts(self):
        """Diagnostic: log frame counts every 5 seconds while recording."""
        if self._recording:
            self.get_logger().info(
                f'[RealSense] Recording: {self._color_frames} color frames, '
                f'{self._depth_frames} depth frames')
            if self._color_frames == 0 or self._depth_frames == 0:
                self.get_logger().warn(
                    '[RealSense] No frames received! Check if camera topics are being published.')

    # ── Shutdown ─────────────────────────────────────────────────────────────

    def _signal_handler(self, signum, frame):
        self.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    def destroy_node(self):
        if self._shutdown_called:
            return
        self._shutdown_called = True
        if self._recording:
            self._stop_recording()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RealSenseRecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()