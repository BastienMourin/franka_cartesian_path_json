#!/usr/bin/env python3
"""
plot_wrench_poses.py — visualise wrench.json + poses.json from a logger run
============================================================================

Usage:
    python3 plot_wrench_poses.py <run_folder>
    python3 plot_wrench_poses.py <run_folder> --save          # save PNG instead
    python3 plot_wrench_poses.py <run_folder> --ns NS1        # only one arm
    python3 plot_wrench_poses.py                              # interactive picker

<run_folder> is the timestamped directory produced by wrench_logger_node, e.g.
    ~/ros2_data/ignacio_cartesian_pose_json/data/20240501_143022/

Dependencies:
    pip install matplotlib numpy
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

PALETTE = {
    'NS1': {'force': '#00C8FF', 'torque': '#FF6B6B', 'pos': '#A8FF78'},
    'NS2': {'force': '#FFD166', 'torque': '#EF476F', 'pos': '#C77DFF'},
    'bg':  '#0D1117',
    'bg2': '#161B22',
    'grid':'#21262D',
    'text':'#E6EDF3',
    'sub': '#8B949E',
}

matplotlib.rcParams.update({
    'figure.facecolor':  PALETTE['bg'],
    'axes.facecolor':    PALETTE['bg2'],
    'axes.edgecolor':    PALETTE['grid'],
    'axes.labelcolor':   PALETTE['text'],
    'axes.titlecolor':   PALETTE['text'],
    'xtick.color':       PALETTE['sub'],
    'ytick.color':       PALETTE['sub'],
    'grid.color':        PALETTE['grid'],
    'grid.linewidth':    0.5,
    'text.color':        PALETTE['text'],
    'legend.facecolor':  PALETTE['bg2'],
    'legend.edgecolor':  PALETTE['grid'],
    'font.family':       'monospace',
    'font.size':         9,
    'axes.titlesize':    10,
    'axes.labelsize':    9,
    'figure.titlesize':  13,
})


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _stamps_to_seconds(entries: list) -> np.ndarray:
    """Convert list of {header:{stamp:{sec,nanosec}}} to float seconds, zero-based."""
    if not entries:
        return np.array([])
    t = np.array([e['header']['stamp']['sec'] + e['header']['stamp']['nanosec'] * 1e-9
                  for e in entries])
    return t - t[0]


def load_wrench(path: str) -> dict:
    with open(path) as f:
        raw = json.load(f)
    out = {}
    for ns, entries in raw.items():
        if not entries:
            out[ns] = None
            continue
        t = _stamps_to_seconds(entries)
        fx = np.array([e['wrench']['force']['x']  for e in entries])
        fy = np.array([e['wrench']['force']['y']  for e in entries])
        fz = np.array([e['wrench']['force']['z']  for e in entries])
        tx = np.array([e['wrench']['torque']['x'] for e in entries])
        ty = np.array([e['wrench']['torque']['y'] for e in entries])
        tz = np.array([e['wrench']['torque']['z'] for e in entries])
        out[ns] = dict(t=t, fx=fx, fy=fy, fz=fz, tx=tx, ty=ty, tz=tz,
                       f_mag=np.sqrt(fx**2 + fy**2 + fz**2),
                       t_mag=np.sqrt(tx**2 + ty**2 + tz**2))
    return out


def load_poses(path: str) -> dict:
    with open(path) as f:
        raw = json.load(f)
    out = {}
    for ns, entries in raw.items():
        if not entries:
            out[ns] = None
            continue
        t = _stamps_to_seconds(entries)
        # filter out samples where TF was unavailable
        valid = [e for e in entries if e.get('translation') is not None]
        t_v   = _stamps_to_seconds(valid) if valid else np.array([])
        px = np.array([e['translation']['x'] for e in valid])
        py = np.array([e['translation']['y'] for e in valid])
        pz = np.array([e['translation']['z'] for e in valid])
        qx = np.array([e['rotation']['x']    for e in valid])
        qy = np.array([e['rotation']['y']    for e in valid])
        qz = np.array([e['rotation']['z']    for e in valid])
        qw = np.array([e['rotation']['w']    for e in valid])
        n_total   = len(entries)
        n_invalid = n_total - len(valid)
        out[ns] = dict(t=t_v, px=px, py=py, pz=pz,
                       qx=qx, qy=qy, qz=qz, qw=qw,
                       n_total=n_total, n_invalid=n_invalid)
    return out


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _ax(fig, gs_cell, title, ylabel, grid=True):
    ax = fig.add_subplot(gs_cell)
    ax.set_title(title, pad=4)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('time (s)')
    if grid:
        ax.grid(True, linestyle='--', alpha=0.4)
    return ax


def plot_wrench_section(fig, gs_row, ns: str, data: dict):
    c = PALETTE[ns]
    alpha = 0.85

    # -- Forces --
    ax_f = _ax(fig, gs_row[0], f'{ns}  ·  Force (N)', 'N')
    ax_f.plot(data['t'], data['fx'], color=c['force'],  lw=1.0, alpha=alpha, label='Fx')
    ax_f.plot(data['t'], data['fy'], color=c['torque'], lw=1.0, alpha=alpha, label='Fy')
    ax_f.plot(data['t'], data['fz'], color=c['pos'],    lw=1.0, alpha=alpha, label='Fz')
    ax_f.plot(data['t'], data['f_mag'], color='white',  lw=1.4, alpha=0.6,  label='|F|', ls='--')
    ax_f.legend(loc='upper right', fontsize=8)

    # -- Torques --
    ax_t = _ax(fig, gs_row[1], f'{ns}  ·  Torque (N·m)', 'N·m')
    ax_t.plot(data['t'], data['tx'], color=c['force'],  lw=1.0, alpha=alpha, label='Tx')
    ax_t.plot(data['t'], data['ty'], color=c['torque'], lw=1.0, alpha=alpha, label='Ty')
    ax_t.plot(data['t'], data['tz'], color=c['pos'],    lw=1.0, alpha=alpha, label='Tz')
    ax_t.plot(data['t'], data['t_mag'], color='white',  lw=1.4, alpha=0.6,  label='|T|', ls='--')
    ax_t.legend(loc='upper right', fontsize=8)

    return ax_f, ax_t


def plot_pose_section(fig, gs_row, ns: str, data: dict):
    c = PALETTE[ns]
    alpha = 0.85
    note = (f'  ({data["n_invalid"]} TF-unavailable samples dropped)'
            if data['n_invalid'] else '')

    # -- XYZ --
    ax_p = _ax(fig, gs_row[0], f'{ns}  ·  TCP position in camera frame (m){note}', 'm')
    ax_p.plot(data['t'], data['px'], color=c['force'],  lw=1.0, alpha=alpha, label='X')
    ax_p.plot(data['t'], data['py'], color=c['torque'], lw=1.0, alpha=alpha, label='Y')
    ax_p.plot(data['t'], data['pz'], color=c['pos'],    lw=1.0, alpha=alpha, label='Z')
    ax_p.legend(loc='upper right', fontsize=8)

    # -- Quaternion --
    ax_q = _ax(fig, gs_row[1], f'{ns}  ·  TCP orientation (quaternion)', '')
    ax_q.plot(data['t'], data['qx'], color=c['force'],  lw=1.0, alpha=alpha, label='qx')
    ax_q.plot(data['t'], data['qy'], color=c['torque'], lw=1.0, alpha=alpha, label='qy')
    ax_q.plot(data['t'], data['qz'], color=c['pos'],    lw=1.0, alpha=alpha, label='qz')
    ax_q.plot(data['t'], data['qw'], color='white',     lw=1.4, alpha=0.75,  label='qw')
    ax_q.legend(loc='upper right', fontsize=8)

    return ax_p, ax_q


def plot_3d_trajectory(fig, gs_cell, ns: str, data: dict):
    """3-D scatter of TCP position coloured by time."""
    ax = fig.add_subplot(gs_cell, projection='3d')
    ax.set_facecolor(PALETTE['bg2'])
    ax.set_title(f'{ns}  ·  TCP trajectory (3-D)', pad=4)

    if len(data['t']) == 0:
        ax.text(0, 0, 0, 'no valid TF data', ha='center', color=PALETTE['sub'])
        return ax

    sc = ax.scatter(data['px'], data['py'], data['pz'],
                    c=data['t'], cmap='plasma', s=6, alpha=0.8)
    fig.colorbar(sc, ax=ax, pad=0.08, shrink=0.6, label='time (s)')
    ax.set_xlabel('X (m)', labelpad=4)
    ax.set_ylabel('Y (m)', labelpad=4)
    ax.set_zlabel('Z (m)', labelpad=4)
    ax.tick_params(colors=PALETTE['sub'])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(PALETTE['grid'])
    ax.yaxis.pane.set_edgecolor(PALETTE['grid'])
    ax.zaxis.pane.set_edgecolor(PALETTE['grid'])
    return ax


# ---------------------------------------------------------------------------
# Main plot builder
# ---------------------------------------------------------------------------

def build_figure(wrench: dict, poses: dict, namespaces: list, run_tag: str,
                 show_3d: bool = True):
    active = [ns for ns in namespaces if wrench.get(ns) or poses.get(ns)]
    if not active:
        print('[ERROR] No data to plot.')
        sys.exit(1)

    n_ns    = len(active)
    n_3d    = n_ns if show_3d else 0
    # rows: per-arm wrench (2 cols) + per-arm pose (2 cols) + optional 3D row
    n_rows  = n_ns * 2 + (1 if n_3d else 0)

    fig = plt.figure(figsize=(16, 4.5 * n_rows))
    fig.suptitle(f'Franka Dual-Arm Log  ·  {run_tag}', y=1.0, fontweight='bold')

    outer = gridspec.GridSpec(n_rows, 1, figure=fig, hspace=0.55)

    row = 0
    for ns in active:
        # wrench row
        inner_w = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[row], wspace=0.35)
        if wrench.get(ns):
            plot_wrench_section(fig, inner_w, ns, wrench[ns])
        else:
            ax = fig.add_subplot(inner_w[0])
            ax.text(0.5, 0.5, f'{ns}: no wrench data', ha='center', va='center',
                    transform=ax.transAxes, color=PALETTE['sub'])
        row += 1

        # pose row
        inner_p = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[row], wspace=0.35)
        if poses.get(ns):
            plot_pose_section(fig, inner_p, ns, poses[ns])
        else:
            ax = fig.add_subplot(inner_p[0])
            ax.text(0.5, 0.5, f'{ns}: no pose data', ha='center', va='center',
                    transform=ax.transAxes, color=PALETTE['sub'])
        row += 1

    # 3-D trajectories
    if n_3d:
        inner_3d = gridspec.GridSpecFromSubplotSpec(1, n_ns, subplot_spec=outer[row],
                                                    wspace=0.45)
        for i, ns in enumerate(active):
            if poses.get(ns):
                plot_3d_trajectory(fig, inner_3d[i], ns, poses[ns])

    fig.patch.set_facecolor(PALETTE['bg'])
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def pick_run_folder_interactively() -> Path:
    """If no argument given, try common data locations and let user pick."""
    candidates = [
        Path.home() / 'ros2_data' / 'ignacio_cartesian_pose_json' / 'data',
    ]
    # also check install/share path pattern
    for p in candidates:
        if p.exists():
            runs = sorted(p.iterdir(), reverse=True)
            if runs:
                print(f'Found runs in {p}:')
                for i, r in enumerate(runs[:10]):
                    print(f'  [{i}] {r.name}')
                idx = input('Select run index [0]: ').strip() or '0'
                return runs[int(idx)]
    print('[ERROR] No run folder found. Pass the path as an argument.')
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Plot wrench + pose JSON logs.')
    parser.add_argument('run_folder', nargs='?', default=None,
                        help='Path to timestamped run folder (contains wrench.json / poses.json)')
    parser.add_argument('--save', action='store_true',
                        help='Save figure as PNG instead of showing interactively')
    parser.add_argument('--ns', nargs='+', default=['NS1', 'NS2'],
                        help='Namespaces to plot (default: NS1 NS2)')
    parser.add_argument('--no3d', action='store_true',
                        help='Skip 3-D trajectory plots')
    args = parser.parse_args()

    run_folder = Path(args.run_folder) if args.run_folder else pick_run_folder_interactively()

    wrench_path = run_folder / 'wrench.json'
    poses_path  = run_folder / 'poses.json'

    if not wrench_path.exists() and not poses_path.exists():
        print(f'[ERROR] Neither wrench.json nor poses.json found in {run_folder}')
        sys.exit(1)

    wrench = load_wrench(str(wrench_path)) if wrench_path.exists() else {}
    poses  = load_poses(str(poses_path))   if poses_path.exists()  else {}

    run_tag = run_folder.name
    fig = build_figure(wrench, poses, args.ns, run_tag, show_3d=not args.no3d)

    if args.save:
        out = run_folder / f'plot_{run_tag}.png'
        fig.savefig(str(out), dpi=150, bbox_inches='tight',
                    facecolor=PALETTE['bg'])
        print(f'Saved → {out}')
    else:
        plt.show()


if __name__ == '__main__':
    main()