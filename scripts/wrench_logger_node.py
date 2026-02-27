#!/usr/bin/env python3
"""
wrench_logger_node.py — Franka dual-arm wrench + TCP pose logger
=================================================================

Save path priority:
  1. ROS param  data_dir   — set directly by the C++ launcher (e.g. .../20240501_143022/wrench/)
  2. Topic      /recording_manager/run_folder  — appends /wrench/ automatically
  3. Fallback   hardcoded root / timestamped subfolder

Parameters:
  data_dir            str     ""
  save_interval_sec   float   5.0
  tf_timeout_sec      float   0.05
  camera_frame        str     camera_color_optical_frame
"""

import json
import os
import signal
from datetime import datetime

import rclpy
import rclpy.time
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import String
from tf2_ros import (
    Buffer, TransformListener,
    LookupException, ExtrapolationException, ConnectivityException,
)

_TF_EXC = (LookupException, ExtrapolationException, ConnectivityException)
_FALLBACK_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'records'
)


def _stamp_dict(stamp):
    return {'sec': stamp.sec, 'nanosec': stamp.nanosec}


class WrenchLoggerNode(Node):

    def __init__(self):
        super().__init__('wrench_logger_node')
        self._shutdown_requested = False
        self._shutdown_called    = False
        self._run_dir: str | None = None
        self.wrench_data: dict | None = None
        self.poses_data:  dict | None = None

        # ── Parameters ───────────────────────────────────────────────────────
        self.declare_parameter('data_dir',          '')
        self.declare_parameter('save_interval_sec', 5.0)
        self.declare_parameter('tf_timeout_sec',    0.05)
        self.declare_parameter('camera_frame',      'camera_color_optical_frame')

        data_dir       = self.get_parameter('data_dir').get_parameter_value().string_value
        save_interval  = self.get_parameter('save_interval_sec').get_parameter_value().double_value
        tf_timeout_sec = self.get_parameter('tf_timeout_sec').get_parameter_value().double_value
        self.camera_frame = self.get_parameter('camera_frame').get_parameter_value().string_value
        self.tf_timeout   = Duration(seconds=tf_timeout_sec)

        # ── Resolve save directory ────────────────────────────────────────────
        if data_dir:
            # Passed directly from C++ node — use as-is
            self._init_run_dir(data_dir)
        else:
            # Subscribe to latched run_folder topic
            latched = QoSProfile(
                depth=1, history=HistoryPolicy.KEEP_LAST,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                reliability=ReliabilityPolicy.RELIABLE)
            self._folder_sub = self.create_subscription(
                String, '/recording_manager/run_folder',
                self._on_run_folder, latched)
            self.get_logger().info(
                'data_dir not set — waiting for /recording_manager/run_folder ...')
            # Also set a one-shot timer to fall back to default if topic never arrives
            self._fallback_timer = self.create_timer(10.0, self._fallback_init)

        # ── TF2 ──────────────────────────────────────────────────────────────
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ── Wrench subscribers ───────────────────────────────────────────────
        self._tcp = {'NS1': 'NS1_fr3_hand_tcp', 'NS2': 'NS2_fr3_hand_tcp'}
        for ns, topic in {
            'NS1': '/NS1/franka_robot_state_broadcaster/external_wrench_in_stiffness_frame',
            'NS2': '/NS2/franka_robot_state_broadcaster/external_wrench_in_stiffness_frame',
        }.items():
            self.create_subscription(WrenchStamped, topic,
                lambda msg, n=ns: self._wrench_cb(msg, n), 10)

        # ── Periodic flush ───────────────────────────────────────────────────
        self.create_timer(save_interval, self._flush)

        # ── Signals ──────────────────────────────────────────────────────────
        signal.signal(signal.SIGINT,  self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    # ── Directory setup ───────────────────────────────────────────────────────

    def _init_run_dir(self, path: str):
        """Set the save directory and initialise data buffers."""
        if self._run_dir is not None:
            return  # already initialised
        os.makedirs(path, exist_ok=True)
        self._run_dir = path
        self.wrench_data = {'NS1': [], 'NS2': []}
        self.poses_data  = {'NS1': [], 'NS2': []}
        self.get_logger().info(f'=== Wrench Logger: saving to {self._run_dir} ===')

    def _on_run_folder(self, msg: String):
        """Called when /recording_manager/run_folder is received."""
        if self._run_dir is not None:
            return
        if hasattr(self, '_fallback_timer'):
            self._fallback_timer.cancel()
        save_dir = os.path.join(msg.data, 'wrench')
        self._init_run_dir(save_dir)

    def _fallback_init(self):
        """Use default path if run_folder topic never arrived after 10 s."""
        self._fallback_timer.cancel()
        if self._run_dir is not None:
            return
        ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = os.path.join(_FALLBACK_ROOT, ts, 'wrench')
        self.get_logger().warn(
            f'run_folder topic not received — using fallback: {save_dir}')
        self._init_run_dir(save_dir)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _wrench_cb(self, msg: WrenchStamped, ns: str):
        if self.wrench_data is None:
            return

        stamp = msg.header.stamp
        child = self._tcp[ns]

        self.wrench_data[ns].append({
            'stamp': _stamp_dict(stamp),
            'frame_id': child,
            'force':  {'x': msg.wrench.force.x,  'y': msg.wrench.force.y,  'z': msg.wrench.force.z},
            'torque': {'x': msg.wrench.torque.x, 'y': msg.wrench.torque.y, 'z': msg.wrench.torque.z},
        })

        # TF pose
        try:
            tf = self.tf_buffer.lookup_transform(
                self.camera_frame, child,
                rclpy.time.Time(), timeout=self.tf_timeout)
            t, r = tf.transform.translation, tf.transform.rotation
            self.poses_data[ns].append({
                'stamp': _stamp_dict(stamp),
                'child_frame': child,
                'parent_frame': self.camera_frame,
                'translation': {'x': t.x, 'y': t.y, 'z': t.z},
                'rotation':    {'x': r.x, 'y': r.y, 'z': r.z, 'w': r.w},
            })
        except _TF_EXC as e:
            self.get_logger().warn(f'[{ns}] TF: {e}', throttle_duration_sec=5.0)

    # ── Flush ─────────────────────────────────────────────────────────────────

    def _flush(self):
        if self.wrench_data is None or self._run_dir is None:
            return
        try:
            with open(os.path.join(self._run_dir, 'wrench.json'), 'w') as f:
                json.dump(self.wrench_data, f, indent=2)
            with open(os.path.join(self._run_dir, 'poses.json'), 'w') as f:
                json.dump(self.poses_data, f, indent=2)
            self.get_logger().info(
                f'Flushed wrench (NS1={len(self.wrench_data["NS1"])} '
                f'NS2={len(self.wrench_data["NS2"])})')
        except Exception as e:
            self.get_logger().error(f'Flush error: {e}')

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def _signal_handler(self, signum, frame):
        self._shutdown_requested = True

    def destroy_node(self):
        if self._shutdown_called:
            return
        self._shutdown_called = True
        self.get_logger().info('Final flush before shutdown...')
        self._flush()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = WrenchLoggerNode()
    try:
        while rclpy.ok() and not node._shutdown_requested:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()