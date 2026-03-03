#!/usr/bin/env python3
"""
recording_manager_node.py — unified start/stop for all recorders
=================================================================

Manages three recording nodes:
  1. wrench_logger_node      — wrench + TCP pose → JSON  (subprocess)
  2. obs_recorder_node       — OBS video          (service call)
  3. realsense_recorder_node — RGB + depth MP4    (service call)

The save path for every recorder is derived from the waypoints file location,
published by the C++ trajectory node on /recording_manager/run_folder:

  waypoints_file = /test/action_profiles_displacement.json
  run_folder     = /test/20240501_143022/
      wrench_poses/← wrench_logger writes here
      obs/         ← obs_recorder writes here
      realsense/   ← realsense_recorder writes here

The manager subscribes to that topic and passes the subfolder paths to each
recorder. It also exposes start/stop services so the operator (or the C++ node)
can control all recorders with a single call.

Services:
  /recording_manager/start   (std_srvs/srv/Trigger)
  /recording_manager/stop    (std_srvs/srv/Trigger)

Status:
  /recording_manager/status  (std_msgs/msg/String)

Parameters (--ros-args -p key:=value):
  auto_record               bool    false
  enable_wrench             bool    true
  enable_obs                bool    true
  enable_realsense          bool    true
    copy_obs_from_default     bool    true    copy OBS output from /home/franka2/obs
  wrench_init_timeout       float   3.0     s to wait for wrench_logger init
  obs_service_timeout       float   5.0     s to wait for OBS service
  realsense_service_timeout float   5.0     s to wait for RealSense service

Usage:
  # Terminal 1 — OBS recorder
  ros2 run ignacio_cartesian_pose_json obs_recorder_node

  # Terminal 2 — RealSense recorder
  ros2 run ignacio_cartesian_pose_json realsense_recorder_node

  # Terminal 3 — Manager (waits for /recording_manager/run_folder then auto-starts)
  ros2 run ignacio_cartesian_pose_json recording_manager_node \\
      --ros-args -p auto_record:=true

  # Disable individual recorders:
  ros2 run ignacio_cartesian_pose_json recording_manager_node \\
      --ros-args -p enable_obs:=false -p auto_record:=true

  # Manual trigger:
  ros2 service call /recording_manager/start std_srvs/srv/Trigger {}
  ros2 service call /recording_manager/stop  std_srvs/srv/Trigger {}
"""

import os
import signal
import shutil
import subprocess
import sys
import time
import fcntl
from pathlib import Path
from datetime import datetime, timedelta

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String
from std_srvs.srv import Trigger

WRENCH_LOGGER_SCRIPT = (
    '/home/franka2/ignacio_ws/src/ignacio_cartesian_pose_json/'
    'scripts/wrench_logger_node.py'
)
OBS_DEFAULT_OUTPUT = '/home/franka2/obs'


class RecordingManagerNode(Node):

    def __init__(self):
        super().__init__('recording_manager_node')
        self._lock_file = None
        self._acquire_process_lock()
        self._shutdown_called = False
        self._recording       = False
        self._wrench_proc: subprocess.Popen | None = None
        self._run_folder: str | None = None   # received from C++ node topic
        self._obs_existing_files: set[str] = set()
        self._recording_start_time: datetime | None = None

        # ── Parameters ───────────────────────────────────────────────────────
        self.declare_parameter('auto_record',               False)
        self.declare_parameter('enable_wrench',             True)
        self.declare_parameter('enable_obs',                True)
        self.declare_parameter('enable_realsense',          True)
        self.declare_parameter('copy_obs_from_default',     True)
        self.declare_parameter('wrench_init_timeout',       3.0)
        self.declare_parameter('obs_service_timeout',       5.0)
        self.declare_parameter('realsense_service_timeout', 5.0)

        self._auto_record  = self.get_parameter('auto_record').get_parameter_value().bool_value
        self._en_wrench    = self.get_parameter('enable_wrench').get_parameter_value().bool_value
        self._en_obs       = self.get_parameter('enable_obs').get_parameter_value().bool_value
        self._en_realsense = self.get_parameter('enable_realsense').get_parameter_value().bool_value
        self._copy_obs_from_default = self.get_parameter('copy_obs_from_default').get_parameter_value().bool_value
        self._t_wrench     = self.get_parameter('wrench_init_timeout').get_parameter_value().double_value
        self._t_obs        = self.get_parameter('obs_service_timeout').get_parameter_value().double_value
        self._t_realsense  = self.get_parameter('realsense_service_timeout').get_parameter_value().double_value

        # ── Subscribe to run_folder (latched — works even if C++ already published) ──
        latched = QoSProfile(
            depth=1, history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(
            String, '/recording_manager/run_folder',
            self._on_run_folder, latched)

        # ── Service clients ───────────────────────────────────────────────────
        self._obs_start = self.create_client(Trigger, 'obs_recorder/start')
        self._obs_stop  = self.create_client(Trigger, 'obs_recorder/stop')
        self._rs_start  = self.create_client(Trigger, 'realsense_recorder/start')
        self._rs_stop   = self.create_client(Trigger, 'realsense_recorder/stop')

        # ── Publisher / services ─────────────────────────────────────────────
        self._status_pub = self.create_publisher(String, 'recording_manager/status', 10)
        self.create_timer(1.0, self._publish_status)
        self.create_service(Trigger, 'recording_manager/start', self._handle_start)
        self.create_service(Trigger, 'recording_manager/stop',  self._handle_stop)

        # ── Signals ───────────────────────────────────────────────────────────
        signal.signal(signal.SIGINT,  self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        enabled = '+'.join(
            [k for k, v in [('wrench', self._en_wrench),
                             ('obs',    self._en_obs),
                             ('realsense', self._en_realsense)] if v])
        self.get_logger().info('recording_manager_node ready.')
        self.get_logger().info(f'  Enabled recorders: {enabled}')
        self.get_logger().info('  Waiting for /recording_manager/run_folder from C++ node...')

        # auto_record fires once run_folder is received (see _on_run_folder)

    # ── Run-folder callback ───────────────────────────────────────────────────

    def _acquire_process_lock(self):
        lock_path = '/tmp/ignacio_recording_manager.lock'
        lock_file = open(lock_path, 'w')
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            lock_file.write(str(os.getpid()))
            lock_file.flush()
            self._lock_file = lock_file
        except BlockingIOError:
            lock_file.close()
            self.get_logger().error(
                'Another recording_manager_node instance is already running. Exiting.')
            raise RuntimeError('recording_manager_node already running')

    def _on_run_folder(self, msg: String):
        """Received from the C++ trajectory node — holds the session root dir."""
        self._run_folder = msg.data
        self.get_logger().info(f'Run folder received: {self._run_folder}')
        self.get_logger().info(f'  wrench_poses/→ {self._run_folder}/wrench_poses/')
        self.get_logger().info(f'  obs/       → {self._run_folder}/obs/')
        self.get_logger().info(f'  realsense/ → {self._run_folder}/realsense/')

        if self._auto_record and not self._recording:
            self.get_logger().info('auto_record=true — starting all recorders.')
            ok, msg_str = self._start_recording()
            if not ok:
                self.get_logger().error(f'auto_record failed: {msg_str}')

    # ── Wrench logger subprocess ──────────────────────────────────────────────

    def _start_wrench(self) -> tuple[bool, str]:
        if not self._en_wrench:
            return True, 'wrench disabled.'
        if self._wrench_proc and self._wrench_proc.poll() is None:
            return False, 'wrench_logger already running.'
        if not os.path.exists(WRENCH_LOGGER_SCRIPT):
            return False, f'Script not found: {WRENCH_LOGGER_SCRIPT}'

        # Pass wrench_poses subfolder directly so the logger doesn't need the topic
        wrench_dir = os.path.join(self._run_folder, 'wrench_poses') if self._run_folder else ''
        cmd = [sys.executable, WRENCH_LOGGER_SCRIPT]
        if wrench_dir:
            cmd += ['--ros-args', '-p', f'data_dir:={wrench_dir}']

        try:
            self._wrench_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.get_logger().info(
                f'wrench_logger started (PID {self._wrench_proc.pid}) → {wrench_dir}')
            time.sleep(self._t_wrench)
            if self._wrench_proc.poll() is not None:
                err = self._wrench_proc.stderr.read().decode(errors='replace')
                return False, f'wrench_logger crashed: {err[:200]}'
            return True, f'wrench_logger running (PID {self._wrench_proc.pid}).'
        except Exception as e:
            return False, f'Failed to launch wrench_logger: {e}'

    def _stop_wrench(self) -> tuple[bool, str]:
        if not self._en_wrench:
            return True, 'wrench disabled.'
        if self._wrench_proc is None:
            return False, 'wrench_logger not running.'
        if self._wrench_proc.poll() is not None:
            self._wrench_proc = None
            return False, 'wrench_logger already exited.'
        pid = self._wrench_proc.pid
        self.get_logger().info(f'SIGINT → wrench_logger (PID {pid}) — flushing JSON...')
        self._wrench_proc.send_signal(signal.SIGINT)
        try:
            self._wrench_proc.wait(timeout=5.0)
            self.get_logger().info(f'wrench_logger (PID {pid}) exited.')
        except subprocess.TimeoutExpired:
            self.get_logger().warn(f'Timeout — killing wrench_logger (PID {pid}).')
            self._wrench_proc.kill()
            self._wrench_proc.wait()
        self._wrench_proc = None
        return True, 'wrench_logger stopped and flushed.'

    # ── OBS file copy helper ──────────────────────────────────────────────────

    @staticmethod
    def _obs_video_candidates(root: str) -> list[Path]:
        candidates: list[Path] = []
        for ext in ('*.mkv', '*.mp4', '*.mov', '*.flv', '*.m4v'):
            candidates.extend(Path(root).glob(ext))
        return candidates

    def _snapshot_obs_files(self):
        if not os.path.exists(OBS_DEFAULT_OUTPUT):
            self._obs_existing_files = set()
            return
        self._obs_existing_files = {
            str(path.resolve()) for path in self._obs_video_candidates(OBS_DEFAULT_OUTPUT)
        }

    def _copy_obs_file_to_run(self, src_file: str) -> tuple[bool, str]:
        if not self._run_folder:
            return False, 'Run folder is not set.'
        src_path = Path(src_file)
        if not src_path.exists():
            return False, f'OBS source file does not exist: {src_file}'

        obs_dest = os.path.join(self._run_folder, 'obs')
        os.makedirs(obs_dest, exist_ok=True)

        dest_path = os.path.join(obs_dest, src_path.name)
        if str(src_path.parent.resolve()) == str(Path(obs_dest).resolve()):
            return True, f'OBS file already in destination: {src_path.name}'

        self.get_logger().info(f'Copying OBS file: {src_path} → {dest_path}')
        shutil.copy2(src_path, dest_path)
        return True, f'OBS file copied: {src_path.name}'

    @staticmethod
    def _extract_obs_file_from_stop_message(msg: str) -> str | None:
        # Expected: "Stopped → /path/to/file.ext"
        if '→' not in msg:
            return None
        path_part = msg.split('→', 1)[1].strip()
        if not path_part:
            return None
        if path_part.endswith(('/', '\\')):
            return None
        return path_part

    def _find_new_obs_file_since_start(self) -> str | None:
        if not os.path.exists(OBS_DEFAULT_OUTPUT):
            return None

        now_candidates = self._obs_video_candidates(OBS_DEFAULT_OUTPUT)
        new_files = []
        for path in now_candidates:
            resolved = str(path.resolve())
            if resolved in self._obs_existing_files:
                continue
            try:
                mtime = datetime.fromtimestamp(path.stat().st_mtime)
            except Exception:
                continue
            if self._recording_start_time is not None and mtime + timedelta(seconds=2) < self._recording_start_time:
                continue
            new_files.append((mtime, resolved))

        if not new_files:
            return None
        new_files.sort(key=lambda item: item[0])
        return new_files[-1][1]

    def _copy_latest_obs_file(self) -> tuple[bool, str]:
        """Copy latest OBS recording from default location to records/obs/."""
        if not self._en_obs or not self._run_folder:
            return True, 'OBS copy skipped.'

        obs_dest = os.path.join(self._run_folder, 'obs')
        if not os.path.exists(OBS_DEFAULT_OUTPUT):
            return False, f'OBS output dir not found: {OBS_DEFAULT_OUTPUT}'

        try:
            # Ensure destination directory exists
            os.makedirs(obs_dest, exist_ok=True)
            
            # Find latest OBS video (created within last 10 minutes)
            now = datetime.now()
            latest_file = None
            latest_time = None

            candidates = self._obs_video_candidates(OBS_DEFAULT_OUTPUT)

            for f in candidates:
                mtime = datetime.fromtimestamp(f.stat().st_mtime)
                if now - mtime < timedelta(minutes=10):
                    if latest_time is None or mtime > latest_time:
                        latest_file = f
                        latest_time = mtime

            if latest_file is None:
                return False, f'No recent OBS video found in {OBS_DEFAULT_OUTPUT}'
            return self._copy_obs_file_to_run(str(latest_file))
        except Exception as e:
            return False, f'OBS copy failed: {e}'

    # ── Generic service call helper ───────────────────────────────────────────

    def _call(self, client, label: str, timeout: float) -> tuple[bool, str]:
        if not client.wait_for_service(timeout_sec=timeout):
            msg = f'[{label}] not available after {timeout}s.'
            self.get_logger().error(msg)
            return False, msg
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if not future.done():
            return False, f'[{label}] call timed out.'
        result = future.result()
        if result is None:
            return False, f'[{label}] returned None.'
        log = self.get_logger().info if result.success else self.get_logger().warn
        log(f'[{label}] {result.message}')
        return result.success, result.message

    # ── Combined start / stop ─────────────────────────────────────────────────

    def _start_recording(self) -> tuple[bool, str]:
        if self._recording:
            return False, 'Already recording.'
        if self._run_folder is None:
            return False, 'Run folder not yet received — waiting for C++ node.'

        self._recording_start_time = datetime.now()
        if self._en_obs and self._copy_obs_from_default:
            self._snapshot_obs_files()

        results = []

        # 1. Wrench — failure is logged but does not block others
        ok, msg = self._start_wrench()
        results.append(f'[wrench] {msg}')
        if not ok:
            self.get_logger().warn(f'wrench start failed: {msg}')

        # 2. OBS — failure is logged but does not block others
        if self._en_obs:
            ok, msg = self._call(self._obs_start, 'obs/start', self._t_obs)
            results.append(f'[obs] {msg}')
            if not ok:
                self.get_logger().warn(f'OBS start failed: {msg}')

        # 3. RealSense — failure is logged but does not block others
        if self._en_realsense:
            ok, msg = self._call(self._rs_start, 'realsense/start', self._t_realsense)
            results.append(f'[realsense] {msg}')
            if not ok:
                self.get_logger().warn(f'RealSense start failed: {msg}')

        self._recording = True
        self.get_logger().info(
            f'=== ALL RECORDERS STARTED | run folder: {self._run_folder} ===')
        return True, ' | '.join(results)

    def _stop_recording(self, force: bool = False) -> tuple[bool, str]:
        if not self._recording and not force:
            return False, 'Not recording.'

        self._recording = False
        results = []

        # Stop in reverse order: realsense → obs → wrench
        # Each stop is attempted unconditionally regardless of individual failures
        if self._en_realsense:
            _, msg = self._call(self._rs_stop, 'realsense/stop', self._t_realsense)
            results.append(f'[realsense] {msg}')

        obs_stop_msg = ''
        if self._en_obs:
            _, msg = self._call(self._obs_stop, 'obs/stop', self._t_obs)
            obs_stop_msg = msg
            results.append(f'[obs] {msg}')

        _, msg = self._stop_wrench()
        results.append(f'[wrench] {msg}')

        # OBS already writes directly into run_folder/obs via data_dir.
        # Optional legacy fallback copy can be enabled via parameter.
        if self._copy_obs_from_default:
            time.sleep(3.0)

            ok = False
            copy_msg = ''

            stop_file = self._extract_obs_file_from_stop_message(obs_stop_msg)
            if stop_file:
                ok, copy_msg = self._copy_obs_file_to_run(stop_file)

            if not ok:
                new_file = self._find_new_obs_file_since_start()
                if new_file:
                    ok, copy_msg = self._copy_obs_file_to_run(new_file)

            if not ok:
                ok, copy_msg = self._copy_latest_obs_file()

            if ok:
                results.append(f'[obs_copy] {copy_msg}')
            else:
                results.append(f'[obs_copy_WARN] {copy_msg}')

        self.get_logger().info('=== ALL RECORDERS STOPPED ===')
        return True, ' | '.join(results)

    # ── Service handlers ──────────────────────────────────────────────────────

    def _handle_start(self, request, response):
        response.success, response.message = self._start_recording()
        return response

    def _handle_stop(self, request, response):
        response.success, response.message = self._stop_recording()
        return response

    # ── Status ────────────────────────────────────────────────────────────────

    def _publish_status(self):
        # Watch for unexpected wrench death
        if self._wrench_proc and self._wrench_proc.poll() is not None:
            self.get_logger().warn('wrench_logger died unexpectedly.')
            self._wrench_proc = None
            if self._recording:
                self._recording = False

        enabled = '+'.join([k for k, v in [
            ('wrench', self._en_wrench), ('obs', self._en_obs),
            ('realsense', self._en_realsense)] if v])
        state = 'RECORDING' if self._recording else 'idle'
        folder = self._run_folder or 'waiting...'
        msg = String()
        msg.data = f'{state} | enabled={enabled} | folder={folder}'
        self._status_pub.publish(msg)

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def _signal_handler(self, signum, frame):
        self.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    def destroy_node(self):
        if self._shutdown_called:
            return
        self._shutdown_called = True
        if self._recording:
            self.get_logger().info('Shutdown: stopping all recorders...')
            self._stop_recording(force=True)
        elif self._wrench_proc:
            self._stop_wrench()
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
        node = RecordingManagerNode()
    except RuntimeError:
        if rclpy.ok():
            rclpy.shutdown()
        return
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