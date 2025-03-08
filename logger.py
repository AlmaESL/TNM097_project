import cv2
import numpy as np
import os
import datetime
from tabulate import tabulate
import matplotlib.pyplot as plt

# ------------------------- Define Directories ------------------------- #
LOG_DIR = "Results"
COMPUTER_LOG_DIR = os.path.join(LOG_DIR, "computer")
PHONE_LOG_DIR = os.path.join(LOG_DIR, "phone")

FRAME_DIR = "Captured_frames"
computer_frames_dir = os.path.join(FRAME_DIR, "computer")
phone_frames_dir = os.path.join(FRAME_DIR, "phone")

DIFF_DIR = "Color difference maps"
computer_diff_dir = os.path.join(DIFF_DIR, "devices")
# phone_diff_dir = os.path.join(DIFF_DIR, "phone")

# Ensure directories exist
for directory in [COMPUTER_LOG_DIR, PHONE_LOG_DIR, 
                  computer_frames_dir, phone_frames_dir, 
                  computer_diff_dir]:
    os.makedirs(directory, exist_ok=True)




# ------------------------- Directory Retrieval Functions ------------------------- #
def get_computer_frames_dir():
    return computer_frames_dir

def get_phone_frames_dir():
    return phone_frames_dir

def get_computer_diff_dir():
    return computer_diff_dir

# def get_phone_diff_dir():
#     return phone_diff_dir


# ------------------------- Logging Functions ------------------------- #
def get_log_file_path(log_dir, device):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(log_dir, f"{device}_log_{timestamp}.txt")


def log_results(log_dir, device, std, avg_diff, max_diff, max_pos, niqe, brisque, paq2piq, nima, piqe):
    """Logs evaluation results in a structured format with a timestamped filename."""
    log_file = get_log_file_path(log_dir, device)

    with open(log_file, 'a', encoding="utf-8") as f:
        f.write("\n" + "="*60 + "\n")
        f.write(f"{device} Evaluation Results - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n")

        table_data = [
            ["Metric", "Value"],
            ["Avg STD - Graininess", f"{std:.3f}"],
            ["Avg Color Diff", f"{avg_diff:.3f}"],
            ["Max Color Diff", f"{max_diff:.3f} (at {max_pos})"],
            ["NIQE", f"{niqe:.3f}"],
            ["PIQE", f"{piqe:.3f}"],
            ["BRISQUE", f"{brisque:.3f}"],
            ["NIMA", f"{nima:.3f}"],
            ["PAQ2PIQ", f"{paq2piq:.3f}"]
        ]

        f.write(tabulate(table_data, headers="firstrow", tablefmt='grid'))
        f.write("\n\n")

def log_computer_results(std, avg_diff, max_diff, max_pos, niqe, brisque, paq2piq, nima, piqe):
    """Logs results for the computer camera."""
    log_results(COMPUTER_LOG_DIR, "Computer", std, avg_diff, max_diff, max_pos, niqe, brisque, paq2piq, nima, piqe)

def log_phone_results(std, avg_diff, max_diff, max_pos, niqe, brisque, paq2piq, nima, piqe):
    """Logs results for the phone camera."""
    log_results(PHONE_LOG_DIR, "Phone", std, avg_diff, max_diff, max_pos, niqe, brisque, paq2piq, nima, piqe)


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
        f"PIQE: {stats['piqe']:.3f}",
        f"BRISQUE: {stats['brisque']:.3f}",
        f"NIMA: {stats['nima']:.3f}",
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
        f"PIQE: {stats['piqe']:.3f}",
        f"BRISQUE: {stats['brisque']:.3f}",
        f"NIMA: {stats['nima']:.3f}",
        f"PaQ2PiQ: {stats['paq2piq']:.3f}"
    ]
    
    y_offset = 30
    for i, line in enumerate(metrics_list):
        cv2.putText(stats_frame, line, (20, y_offset + i*30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 100), 1, cv2.LINE_AA)
        
    
    cv2.imshow("Phone Stats Window", stats_frame)



# ------------------------- Color Difference Map Saving ------------------------- #
def save_color_difference_maps(diff_maps, save_dir, device):
    """Saves color difference maps as images with color bars."""
    for i, diff_map in enumerate(diff_maps):
        plt.figure(figsize=(6, 6))
        plt.imshow(diff_map, cmap='viridis')
        plt.colorbar(label='Color Difference')
        plt.axis('off')

        save_path = os.path.join(save_dir, f"{device}_diff_{i}.png")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()

    print(f"Saved {len(diff_maps)} color difference maps to directory")