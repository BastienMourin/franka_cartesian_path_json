#!/usr/bin/env python3
"""
obs_recorder_node.py — ROS 2 node that controls OBS Studio recording
=====================================================================

Save path priority:
  1. ROS param  data_dir   — absolute path used directly
  2. Topic      /recording_manager/run_folder  — appends /obs/ automatically
  3. Fallback   default root / timestamped folder if neither arrives

Services:
  /obs_recorder/start   (std_srvs/srv/Trigger)
  /obs_recorder/stop    (std_srvs/srv/Trigger)

Status:
  /obs_recorder/status  (std_msgs/msg/String)

Parameters:
  obs_host        str     localhost
  obs_port        int     4455
  obs_password    str     ""
  data_dir        str     ""
  auto_record     bool    false
  status_rate_hz  float   1.0

Dependencies:  pip install obsws-python
OBS setup:     Tools → WebSocket Server Settings → Enable
"""

import os
import signal
import sys
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String
from std_srvs.srv import Trigger

try:
    import obsws_python as obs
except ImportError:
    print('[obs_recorder_node] ERROR: pip install obsws-python')
    sys.exit(1)

_FALLBACK_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'records'
)


class ObsRecorderNode(Node):

    def __init__(self):
        super().__init__('obs_recorder_node')
        self._shutdown_called = False
        self._recording       = False
        self._obs             = None
        self._save_dir: str | None = None

        # ── Parameters ───────────────────────────────────────────────────────
        self.declare_parameter('obs_host',       'localhost')
        self.declare_parameter('obs_port',       4455)
        self.declare_parameter('obs_password',   '')
        self.declare_parameter('data_dir',       '')
        self.declare_parameter('auto_record',    False)
        self.declare_parameter('status_rate_hz', 1.0)

        host        = self.get_parameter('obs_host').get_parameter_value().string_value
        port        = self.get_parameter('obs_port').get_parameter_value().integer_value
        password    = self.get_parameter('obs_password').get_parameter_value().string_value
        data_dir    = self.get_parameter('data_dir').get_parameter_value().string_value
        auto_record = self.get_parameter('auto_record').get_parameter_value().bool_value
        rate_hz     = self.get_parameter('status_rate_hz').get_parameter_value().double_value

        # ── Resolve save dir ─────────────────────────────────────────────────
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

        # ── Connect to OBS ───────────────────────────────────────────────────
        self.get_logger().info(f'Connecting to OBS at {host}:{port} ...')
        try:
            self._obs = obs.ReqClient(
                host=host, port=port,
                password=password if password else None, timeout=5)
            self.get_logger().info('Connected to OBS WebSocket server.')
        except Exception as e:
            self.get_logger().error(f'Failed to connect to OBS: {e}')
            self.get_logger().error(
                'Make sure OBS is running with WebSocket enabled '
                '(Tools → WebSocket Server Settings).')
            self._obs = None

        # ── Publishers / services ─────────────────────────────────────────────
        self._status_pub = self.create_publisher(String, 'obs_recorder/status', 10)
        self.create_timer(1.0 / rate_hz, self._publish_status)
        self.create_service(Trigger, 'obs_recorder/start', self._handle_start)
        self.create_service(Trigger, 'obs_recorder/stop',  self._handle_stop)

        signal.signal(signal.SIGINT,  self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.get_logger().info('obs_recorder_node ready.')
        self.get_logger().info('  /obs_recorder/start  |  /obs_recorder/stop')

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
        self._set_save_dir(os.path.join(msg.data, 'obs'))

    def _fallback_init(self):
        self._fallback_timer.cancel()
        if self._save_dir is not None:
            return
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(_FALLBACK_ROOT, ts, 'obs')
        self.get_logger().warn(f'run_folder topic not received — using fallback: {path}')
        self._set_save_dir(path)

    # ── OBS helpers ──────────────────────────────────────────────────────────

    def _set_obs_output_path(self, path: str):
        if self._obs is None:
            return
        for cat, name in [('SimpleOutput', 'FilePath'), ('Output', 'FilePath')]:
            try:
                self._obs.set_profile_parameter(
                    parameter_category=cat, parameter_name=name, parameter_value=path)
                self.get_logger().info(f'OBS output → [{cat}] = {path}')
            except Exception as e:
                self.get_logger().debug(f'[{cat}] skipped: {e}')

    def _start_recording(self) -> tuple[bool, str]:
        if self._obs is None:
            return False, 'OBS not connected.'
        if self._recording:
            return False, 'Already recording.'
        if self._save_dir is None:
            return False, 'Save dir not yet known (waiting for run_folder).'
        self._set_obs_output_path(self._save_dir)
        try:
            self._obs.start_record()
            self._recording = True
            self.get_logger().info(f'OBS recording STARTED → {self._save_dir}')
            return True, f'Recording started → {self._save_dir}'
        except Exception as e:
            return False, f'OBS start failed: {e}'

    def _stop_recording(self) -> tuple[bool, str]:
        if self._obs is None:
            return False, 'OBS not connected.'
        if not self._recording:
            return False, 'Not recording.'
        try:
            resp = self._obs.stop_record()
            self._recording = False
            saved = getattr(resp, 'output_path', None) or self._save_dir or 'unknown'
            self.get_logger().info(f'OBS recording STOPPED → {saved}')
            return True, f'Stopped → {saved}'
        except Exception as e:
            return False, f'OBS stop failed: {e}'

    def _sync_state(self):
        if self._obs is None:
            return
        try:
            self._recording = self._obs.get_record_status().output_active
        except Exception:
            pass

    # ── Service handlers / status ─────────────────────────────────────────────

    def _handle_start(self, request, response):
        response.success, response.message = self._start_recording()
        return response

    def _handle_stop(self, request, response):
        response.success, response.message = self._stop_recording()
        return response

    def _publish_status(self):
        self._sync_state()
        msg = String()
        msg.data = f'recording → {self._save_dir}' if self._recording else 'idle'
        self._status_pub.publish(msg)

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
        if self._obs:
            try:
                self._obs.disconnect()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ObsRecorderNode()
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