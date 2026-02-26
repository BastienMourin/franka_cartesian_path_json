#!/usr/bin/env python3
"""
ROS2 node — Franka dual-arm wrench + TCP pose logger
=====================================================

Subscribes to:
  /NS1/franka_robot_state_broadcaster/external_wrench_in_stiffness_frame
  /NS2/franka_robot_state_broadcaster/external_wrench_in_stiffness_frame

Looks up via TF2:
  camera_color_optical_frame <- NS1_fr3_hand_tcp
  camera_color_optical_frame <- NS2_fr3_hand_tcp

TF tree (from your setup):
  camera_color_optical_frame
      └── ... <- base <- NS1_fr3_link0 <- ... <- NS1_fr3_hand_tcp
                  └── NS2_base <- NS2_fr3_link0 <- ... <- NS2_fr3_hand_tcp
                      (bridge: base->NS2_base published by your launch file
                       tx=1.6, ty=0.0, tz=0.02, yaw=180°)

NOTE: NS1_fr3_hand_tcp and NS2_fr3_hand_tcp are static transforms
      (buffer_length=0.0 in view_frames), so we ALWAYS use
      rclpy.time.Time() (latest) for lookups — querying at an exact
      message timestamp would always fail for static-only chains.

Writes two JSON files inside a timestamped run folder:
  <PACKAGE_PATH>/data/<YYYYMMDD_HHMMSS>/
      wrench.json   — wrench force/torque for NS1 & NS2
      poses.json    — TCP pose in camera frame for NS1 & NS2

Stop recording: Ctrl+C  (triggers final flush before exit)

Parameters (--ros-args -p key:=value):
  save_interval_sec   float   5.0     Seconds between JSON flushes
  tf_timeout_sec      float   0.05    Max wait for a TF transform
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
from geometry_msgs.msg import WrenchStamped

import signal
import sys

from tf2_ros import (
    Buffer,
    TransformListener,
    LookupException,
    ExtrapolationException,
    ConnectivityException,
)

# All TF exceptions we want to silently handle
_TF_EXCEPTIONS = (LookupException, ExtrapolationException, ConnectivityException)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stamp_to_dict(stamp) -> dict:
    return {'sec': stamp.sec, 'nanosec': stamp.nanosec}


def _transform_to_dict(tf_stamped) -> dict:
    """TransformStamped → plain dict with translation + quaternion."""
    t = tf_stamped.transform.translation
    r = tf_stamped.transform.rotation
    return {
        'parent_frame': tf_stamped.header.frame_id,
        'child_frame':  tf_stamped.child_frame_id,
        'stamp':        _stamp_to_dict(tf_stamped.header.stamp),
        'translation':  {'x': t.x, 'y': t.y, 'z': t.z},
        'rotation':     {'x': r.x, 'y': r.y, 'z': r.z, 'w': r.w},
    }


def _resolve_data_dir() -> str:
    return '/home/franka2/ignacio_ws/src/ignacio_cartesian_pose_json/data/'


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class WrenchLoggerNode(Node):

    def __init__(self):
        super().__init__('wrench_logger_node')
        self._shutdown_requested = False
        self._shutdown_called = False  # guard against double-shutdown

        # -- Parameters ------------------------------------------------------
        self.declare_parameter('save_interval_sec', 5.0)
        self.declare_parameter('tf_timeout_sec',    0.05)
        self.declare_parameter('camera_frame',      'camera_color_optical_frame')

        save_interval     = self.get_parameter('save_interval_sec').get_parameter_value().double_value
        tf_timeout_sec    = self.get_parameter('tf_timeout_sec').get_parameter_value().double_value
        self.camera_frame = self.get_parameter('camera_frame').get_parameter_value().string_value
        self.tf_timeout   = Duration(seconds=tf_timeout_sec)

        # -- Timestamped run folder ------------------------------------------
        run_ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_dir = _resolve_data_dir()
        run_dir  = os.path.join(base_dir, run_ts)
        os.makedirs(run_dir, exist_ok=True)

        self.wrench_path = os.path.join(run_dir, 'wrench.json')
        self.poses_path  = os.path.join(run_dir, 'poses.json')

        # -- Data buffers ----------------------------------------------------
        self.wrench_data: dict[str, list] = {'NS1': [], 'NS2': []}
        self.poses_data:  dict[str, list] = {'NS1': [], 'NS2': []}

        # -- Namespace → TCP child frame (static transforms) -----------------
        self.tcp_frame = {
            'NS1': 'NS1_fr3_hand_tcp',
            'NS2': 'NS2_fr3_hand_tcp',
        }

        # -- TF2 ------------------------------------------------------------
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # -- Wrench subscribers ---------------------------------------------
        topics = {
            'NS1': '/NS1/franka_robot_state_broadcaster/external_wrench_in_stiffness_frame',
            'NS2': '/NS2/franka_robot_state_broadcaster/external_wrench_in_stiffness_frame',
        }
        for ns, topic in topics.items():
            self.create_subscription(
                WrenchStamped,
                topic,
                lambda msg, n=ns: self._wrench_callback(msg, n),
                10,
            )

        # -- Periodic flush timer -------------------------------------------
        self.create_timer(save_interval, self._save_to_json)

        # -- Signal handler so Ctrl+C always flushes ------------------------
        signal.signal(signal.SIGINT,  self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.get_logger().info('=== Wrench Logger Node started ===')
        self.get_logger().info(f'  Run folder : {run_dir}')
        self.get_logger().info(f'  wrench.json: {self.wrench_path}')
        self.get_logger().info(f'  poses.json : {self.poses_path}')
        for ns, topic in topics.items():
            self.get_logger().info(f'  [{ns}] topic     : {topic}')
            self.get_logger().info(
                f'  [{ns}] TF lookup : {self.camera_frame} <- {self.tcp_frame[ns]}'
            )
        self.get_logger().info(
            f'  Flush every {save_interval}s  |  TF timeout {tf_timeout_sec}s'
        )
        self.get_logger().info(
            '  TF note: using latest transform (TCP frames are static, '
            'timestamp-exact lookup would always fail)'
        )
        self.get_logger().info('  Stop recording: Ctrl+C')

    # -----------------------------------------------------------------------
    # Signal handler — flush then exit cleanly
    # -----------------------------------------------------------------------

    # Replace your signal handler with:
    def _signal_handler(self, signum, frame):
        self.get_logger().info(f'Signal {signum} received — shutting down...')
        self._shutdown_requested = True   # ← NO sys.exit()

    # -----------------------------------------------------------------------
    # Wrench callback
    # -----------------------------------------------------------------------

    def _wrench_callback(self, msg: WrenchStamped, namespace: str):
        stamp = msg.header.stamp
        child = self.tcp_frame[namespace]

        wrench_entry = {
            'header': {
                'stamp':    _stamp_to_dict(stamp),
                'frame_id': child,
            },
            'wrench': {
                'force':  {'x': msg.wrench.force.x,
                        'y': msg.wrench.force.y,
                        'z': msg.wrench.force.z},
                'torque': {'x': msg.wrench.torque.x,
                        'y': msg.wrench.torque.y,
                        'z': msg.wrench.torque.z},
            },
        }

        pose_entry = self._lookup_pose(child, stamp, namespace)

        # Only save samples where TF was available
        if pose_entry.get('translation') is not None:
            self.wrench_data[namespace].append(wrench_entry)
            self.poses_data[namespace].append(pose_entry)        
            stamp = msg.header.stamp
            child = self.tcp_frame[namespace]

            # wrench entry
            wrench_entry = {
                'header': {
                    'stamp':    _stamp_to_dict(stamp),
                    'frame_id': child,
                },
                'wrench': {
                    'force':  {'x': msg.wrench.force.x,
                            'y': msg.wrench.force.y,
                            'z': msg.wrench.force.z},
                    'torque': {'x': msg.wrench.torque.x,
                            'y': msg.wrench.torque.y,
                            'z': msg.wrench.torque.z},
                },
            }
            self.wrench_data[namespace].append(wrench_entry)

            # pose entry — always look up latest because TCP is a static transform
            pose_entry = self._lookup_pose(child, stamp, namespace)
            self.poses_data[namespace].append(pose_entry)

    # -----------------------------------------------------------------------
    # TF lookup  — always use latest (Time=0) because the hand_tcp frames
    # are static (buffer_length=0 in view_frames).  An exact-timestamp lookup
    # on a static-only chain will always raise ExtrapolationException.
    # -----------------------------------------------------------------------

    def _lookup_pose(self, child: str, stamp, namespace: str) -> dict:
        base = {
            'header': {
                # Keep the wrench message timestamp for alignment in post-processing
                'stamp':       _stamp_to_dict(stamp),
                'frame_id':    self.camera_frame,
                'child_frame': child,
            },
        }

        try:
            tf_stamped = self.tf_buffer.lookup_transform(
                target_frame=self.camera_frame,
                source_frame=child,
                time=rclpy.time.Time(),   # Time(0) = latest available
                timeout=self.tf_timeout,
            )
            base.update(_transform_to_dict(tf_stamped))
            return base

        except _TF_EXCEPTIONS as e:
            self.get_logger().warn(
                f'[{namespace}] TF {child} -> {self.camera_frame} not available yet: '
                f'{type(e).__name__}: {e}',
                throttle_duration_sec=5.0,
            )
        except Exception as e:
            self.get_logger().error(
                f'[{namespace}] Unexpected TF error: {type(e).__name__}: {e}',
                throttle_duration_sec=5.0,
            )

        # Return null pose so the wrench sample is never silently dropped
        base['translation'] = None
        base['rotation']    = None
        base['_tf_note']    = 'TF unavailable at this sample'
        return base

    # -----------------------------------------------------------------------
    # JSON flush
    # -----------------------------------------------------------------------

    def _save_to_json(self):
        errors = []
        for path, buf, label in [
            (self.wrench_path, self.wrench_data, 'wrench'),
            (self.poses_path,  self.poses_data,  'poses'),
        ]:
            try:
                with open(path, 'w') as f:
                    json.dump(buf, f, indent=2)
            except Exception as e:
                errors.append(f'{label}: {e}')

        n1w = len(self.wrench_data['NS1'])
        n2w = len(self.wrench_data['NS2'])
        n1p = len(self.poses_data['NS1'])
        n2p = len(self.poses_data['NS2'])

        if errors:
            self.get_logger().error(f'Save errors: {"; ".join(errors)}')
        else:
            self.get_logger().info(
                f'Flushed — wrench (NS1={n1w}, NS2={n2w}) | '
                f'poses (NS1={n1p}, NS2={n2p})'
            )

    # -----------------------------------------------------------------------
    # Shutdown — always do a final flush
    # -----------------------------------------------------------------------

    def destroy_node(self):
        if self._shutdown_called:
            return
        self._shutdown_called = True
        self.get_logger().info('Shutting down — final save...')
        self._save_to_json()
        super().destroy_node()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# In main(), replace rclpy.spin() with:
def main(args=None):
    rclpy.init(args=args)
    node = WrenchLoggerNode()
    signal.signal(signal.SIGINT,  node._signal_handler)
    signal.signal(signal.SIGTERM, node._signal_handler)
    try:
        while rclpy.ok() and not node._shutdown_requested:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()