# Install deps (one-time)
pip install matplotlib numpy

# Basic — opens interactive window
python3 plot_wrench_poses.py /path/to/20260226_110027/

# Save as PNG instead
python3 plot_wrench_poses.py /path/to/run_folder/ --save

# Only plot one arm
python3 plot_wrench_poses.py /path/to/run_folder/ --ns NS1

# Skip 3-D trajectory plots
python3 plot_wrench_poses.py /path/to/run_folder/ --no3d

# No argument → interactive picker from common data location
python3 plot_wrench_poses.py