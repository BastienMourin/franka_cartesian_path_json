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
    enforce_max_fps bool    true
    auto_detect_fps bool    true
    fps_warmup_frames int   20
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
import fcntl
from collections import deque
from datetime import datetime

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
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
        self._lock_file = None
        self._acquire_process_lock()
        self._shutdown_called = False
        self._lock            = threading.Lock()
        self._bridge          = CvBridge()

        self._recording        = False
        self._save_dir: str | None = None
        self._writer_color: cv2.VideoWriter | None = None
        self._writer_depth: cv2.VideoWriter | None = None
        self._writer_color_fps: float | None = None
        self._writer_depth_fps: float | None = None
        self._last_color_write_ts: float | None = None
        self._last_depth_write_ts: float | None = None
        self._color_frames = 0
        self._depth_frames = 0
        self._color_buffer: list[np.ndarray] = []
        self._depth_buffer: list[np.ndarray] = []
        self._color_times: deque[float] = deque(maxlen=200)
        self._depth_times: deque[float] = deque(maxlen=200)

        # ── Parameters ───────────────────────────────────────────────────────
        self.declare_parameter('data_dir',       '')
        self.declare_parameter('auto_record',    False)
        self.declare_parameter('fps',            30.0)
        self.declare_parameter('enforce_max_fps', True)
        self.declare_parameter('auto_detect_fps', True)
        self.declare_parameter('fps_warmup_frames', 20)
        self.declare_parameter('rgb_topic',      _DEFAULT_RGB)
        self.declare_parameter('depth_topic',    _DEFAULT_DEPTH)
        self.declare_parameter('status_rate_hz', 1.0)

        data_dir        = self.get_parameter('data_dir').get_parameter_value().string_value
        auto_record     = self.get_parameter('auto_record').get_parameter_value().bool_value
        self._fps       = self.get_parameter('fps').get_parameter_value().double_value
        self._enforce_max_fps = self.get_parameter('enforce_max_fps').get_parameter_value().bool_value
        self._auto_detect_fps = self.get_parameter('auto_detect_fps').get_parameter_value().bool_value
        self._fps_warmup_frames = int(
            self.get_parameter('fps_warmup_frames').get_parameter_value().integer_value)
        if self._fps_warmup_frames < 2:
            self._fps_warmup_frames = 2
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

    def _acquire_process_lock(self):
        lock_path = '/tmp/ignacio_realsense_recorder.lock'
        lock_file = open(lock_path, 'w')
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            lock_file.write(str(os.getpid()))
            lock_file.flush()
            self._lock_file = lock_file
        except BlockingIOError:
            lock_file.close()
            self.get_logger().error(
                'Another realsense_recorder_node instance is already running. Exiting.')
            raise RuntimeError('realsense_recorder_node already running')

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

    @staticmethod
    def _stamp_to_sec(msg: Image) -> float:
        sec = float(msg.header.stamp.sec)
        nsec = float(msg.header.stamp.nanosec)
        return sec + nsec * 1e-9

    def _estimate_fps(self, times: deque[float]) -> float:
        if len(times) < 2:
            return float(self._fps)
        deltas = []
        prev = None
        for t in times:
            if prev is not None:
                dt = t - prev
                if dt > 1e-4:
                    deltas.append(dt)
            prev = t
        if not deltas:
            return float(self._fps)
        mean_dt = sum(deltas) / len(deltas)
        fps = 1.0 / mean_dt if mean_dt > 0.0 else float(self._fps)
        return max(1.0, min(120.0, fps))

    def _open_color_writer_if_ready(self, width: int, height: int):
        if self._writer_color is not None or not self._recording or self._save_dir is None:
            return
        if self._auto_detect_fps and len(self._color_times) < self._fps_warmup_frames:
            return
        fps = self._estimate_fps(self._color_times) if self._auto_detect_fps else float(self._fps)
        path = os.path.join(self._save_dir, 'color.mp4')
        self._writer_color = cv2.VideoWriter(path, self._fourcc, fps, (width, height))
        self._writer_color_fps = fps
        self.get_logger().info(f'Color writer: {path} ({width}x{height}@{fps:.2f}fps)')
        for frame in self._color_buffer:
            self._writer_color.write(frame)
            self._color_frames += 1
        self._color_buffer.clear()

    def _open_depth_writer_if_ready(self, width: int, height: int):
        if self._writer_depth is not None or not self._recording or self._save_dir is None:
            return
        if self._auto_detect_fps and len(self._depth_times) < self._fps_warmup_frames:
            return
        fps = self._estimate_fps(self._depth_times) if self._auto_detect_fps else float(self._fps)
        path = os.path.join(self._save_dir, 'depth.mp4')
        self._writer_depth = cv2.VideoWriter(path, self._fourcc, fps, (width, height))
        self._writer_depth_fps = fps
        self.get_logger().info(f'Depth writer: {path} ({width}x{height}@{fps:.2f}fps)')
        for frame in self._depth_buffer:
            self._writer_depth.write(frame)
            self._depth_frames += 1
        self._depth_buffer.clear()

    def _allow_frame_by_rate(self, stamp_sec: float, last_stamp_sec: float | None) -> bool:
        if not self._enforce_max_fps:
            return True
        if self._fps <= 0.0:
            return True
        if last_stamp_sec is None:
            return True
        min_dt = 1.0 / self._fps
        return (stamp_sec - last_stamp_sec) >= (min_dt * 0.95)

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
            self._writer_color_fps = None
            self._writer_depth_fps = None
            self._writer_color = None   # created lazily on first frame
            self._writer_depth = None
            self._last_color_write_ts = None
            self._last_depth_write_ts = None
            self._color_buffer.clear()
            self._depth_buffer.clear()
            self._color_times.clear()
            self._depth_times.clear()
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

            # Short-run safety: if writer never opened yet, open now and flush buffers.
            if self._writer_color is None and self._color_buffer and self._save_dir is not None:
                h, w = self._color_buffer[0].shape[:2]
                fps = self._estimate_fps(self._color_times) if self._auto_detect_fps else float(self._fps)
                path = os.path.join(self._save_dir, 'color.mp4')
                self._writer_color = cv2.VideoWriter(path, self._fourcc, fps, (w, h))
                self._writer_color_fps = fps
                for frame in self._color_buffer:
                    self._writer_color.write(frame)
                    self._color_frames += 1

            if self._writer_depth is None and self._depth_buffer and self._save_dir is not None:
                h, w = self._depth_buffer[0].shape[:2]
                fps = self._estimate_fps(self._depth_times) if self._auto_detect_fps else float(self._fps)
                path = os.path.join(self._save_dir, 'depth.mp4')
                self._writer_depth = cv2.VideoWriter(path, self._fourcc, fps, (w, h))
                self._writer_depth_fps = fps
                for frame in self._depth_buffer:
                    self._writer_depth.write(frame)
                    self._depth_frames += 1

            if self._writer_color:
                self._writer_color.release()
                self._writer_color = None
            if self._writer_depth:
                self._writer_depth.release()
                self._writer_depth = None
            self._color_buffer.clear()
            self._depth_buffer.clear()
            n_col, n_dep, d = self._color_frames, self._depth_frames, self._save_dir
        msg = (f'Stopped — {n_col} color, {n_dep} depth frames '
               f'(fps color={self._writer_color_fps}, depth={self._writer_depth_fps}) → {d}')
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
            self._color_times.append(self._stamp_to_sec(msg))
            stamp_sec = self._color_times[-1]
            if not self._allow_frame_by_rate(stamp_sec, self._last_color_write_ts):
                return
            if self._writer_color is None:
                h, w = frame.shape[:2]
                self._color_buffer.append(frame)
                self._open_color_writer_if_ready(w, h)
                self._last_color_write_ts = stamp_sec
            else:
                self._writer_color.write(frame)
                self._color_frames += 1
                self._last_color_write_ts = stamp_sec

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
            self._depth_times.append(self._stamp_to_sec(msg))
            stamp_sec = self._depth_times[-1]
            if not self._allow_frame_by_rate(stamp_sec, self._last_depth_write_ts):
                return

            if self._writer_depth is None:
                h, w = disp_bgr.shape[:2]
                self._depth_buffer.append(disp_bgr)
                self._open_depth_writer_if_ready(w, h)
                self._last_depth_write_ts = stamp_sec
            else:
                self._writer_depth.write(disp_bgr)
                self._depth_frames += 1
                self._last_depth_write_ts = stamp_sec

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
                        f'depth={self._depth_frames} '
                        f'fps(c={self._writer_color_fps},d={self._writer_depth_fps}) '
                        f'→ {self._save_dir}')
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
        if self._lock_file:
            try:
                fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
                self._lock_file.close()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    try:
        node = RealSenseRecorderNode()
    except RuntimeError:
        if rclpy.ok():
            rclpy.shutdown()
        return
    try:
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            executor.shutdown()
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()