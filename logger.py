import cv2
import numpy as np
import os
import datetime
from tabulate import tabulate

# Define logging directories
LOG_DIR = "Results"
COMPUTER_LOG_DIR = os.path.join(LOG_DIR, "computer")
PHONE_LOG_DIR = os.path.join(LOG_DIR, "phone")

# Ensure directories exist
os.makedirs(COMPUTER_LOG_DIR, exist_ok=True)
os.makedirs(PHONE_LOG_DIR, exist_ok=True)

# Function to generate time stamped log file paths
def get_log_file_path(log_dir, device):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(log_dir, f"{device}_log_{timestamp}.txt")

# Function to log results
def log_results(log_dir, device, std, avg_diff, max_diff, max_pos, niqe, brisque, paq2piq):
    """
    Logs evaluation results in a structured format with a timestamped filename.

    Parameters
    ----------
    log_dir : str
        Directory where logs should be saved.
    device : str
        The device type (e.g., "Computer" or "Phone").
    std : float
        Average standard deviation (graininess).
    avg_diff : float
        Average color difference.
    max_diff : float
        Maximum observed color difference.
    max_pos : tuple
        Location of the max color difference in the frame.
    niqe : float
        NIQE quality metric.
    brisque : float
        BRISQUE quality metric.
    paq2piq : float
        PaQ2PiQ quality metric.
    """
    log_file = get_log_file_path(log_dir, device)

    with open(log_file, 'a', encoding="utf-8") as f:
        
        # Write header
        f.write("\n" + "="*60 + "\n")
        f.write(f"{device} Evaluation Results - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n")
        
        # Write table of metrics values 
        table_data = [
            ["Metric", "Value"],
            ["Avg STD - Graininess", f"{std:.3f}"],
            ["Avg Color Diff", f"{avg_diff:.3f}"],
            ["Max Color Diff", f"{max_diff:.3f} (at {max_pos})"],
            ["NIQE", f"{niqe:.3f}"],
            ["BRISQUE", f"{brisque:.3f}"],
            ["PAQ2PIQ", f"{paq2piq:.3f}"]
        ]
        
        f.write(tabulate(table_data, headers="firstrow", tablefmt='grid'))
        f.write("\n\n")

# Separate logging functions for computer and phone
def log_computer_results(std, avg_diff, max_diff, max_pos, niqe, brisque, paq2piq):
    """Logs results for the computer camera."""
    log_results(COMPUTER_LOG_DIR, "Computer", std, avg_diff, max_diff, max_pos, niqe, brisque, paq2piq)

def log_phone_results(std, avg_diff, max_diff, max_pos, niqe, brisque, paq2piq):
    """Logs results for the phone camera."""
    log_results(PHONE_LOG_DIR, "Phone", std, avg_diff, max_diff, max_pos, niqe, brisque, paq2piq)
    
    
def draw_stats_window_computer(stats, width=600, height=600): 
    """
    Draws a statistics window displaying various quality metrics.

    Parameters
    ----------
    stats : dict
        Dictionary containing the statistics to display, with keys 'std', 
        'avg_diff', 'max_diff', 'min_diff', 'max_pos', 'min_pos', 'niqe', 
        'brisque', and 'paq2piq', each mapping to their respective values.
    width : int, optional
        The width of the stats window, default is 600.
    height : int, optional
        The height of the stats window, default is 600.

    Returns
    -------
    None
    """

    stats_frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    metrics_list = [
        f"Avg STD - Graininess: {stats['std']:.3f}",
        f"Avg Color Diff: {stats['avg_diff']:.3f}",
        f"Max Color Diff: {stats['max_diff']:.3f} at {stats['max_pos']}",
        f"NIQE: {stats['niqe']:.3f}",
        f"BRISQUE: {stats['brisque']:.3f}",
        f"PaQ2PiQ: {stats['paq2piq']:.3f}"
    ]
    
    y_offset = 30
    for i, line in enumerate(metrics_list):
        cv2.putText(stats_frame, line, (20, y_offset + i*30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 100), 1, cv2.LINE_AA)
        
    
    cv2.imshow("Computer Stats Window", stats_frame)
    
    
    
    
def draw_stats_window_phone(stats, width=600, height=600): 
    """
    Draws a statistics window displaying various quality metrics.

    Parameters
    ----------
    stats : dict
        Dictionary containing the statistics to display, with keys 'std', 
        'avg_diff', 'max_diff', 'min_diff', 'max_pos', 'min_pos', 'niqe', 
        'brisque', and 'paq2piq', each mapping to their respective values.
    width : int, optional
        The width of the stats window, default is 600.
    height : int, optional
        The height of the stats window, default is 600.

    Returns
    -------
    None
    """

    stats_frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    metrics_list = [
        f"Avg STD - Graininess: {stats['std']:.3f}",
        f"Avg Color Diff: {stats['avg_diff']:.3f}",
        f"Max Color Diff: {stats['max_diff']:.3f} at {stats['max_pos']}",
        f"NIQE: {stats['niqe']:.3f}",
        f"BRISQUE: {stats['brisque']:.3f}",
        f"PaQ2PiQ: {stats['paq2piq']:.3f}"
    ]
    
    y_offset = 30
    for i, line in enumerate(metrics_list):
        cv2.putText(stats_frame, line, (20, y_offset + i*30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 100), 1, cv2.LINE_AA)
        
    
    cv2.imshow("Phone Stats Window", stats_frame)
