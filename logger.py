from tabulate import tabulate
import numpy as np
import cv2


def log_computer_results(file_path, std, avg_diff, max_diff, max_pos, niqe, brisque, paq2piq): 
    
    """
    Logs the results of the evaluation to a text file.

    Parameters
    ----------
    file_path : str
        File path to write the results to
    std : float
        Average standard deviation of the frames in the buffer
    avg_diff : float
        Average color difference between consecutive frames in the buffer
    max_diff : float
        Maximum color difference observed between consecutive frames in the buffer
    min_diff : float
        Minimum color difference observed between consecutive frames in the buffer
    max_pos : tuple
        Location of the maximum color difference in the frame
    min_pos : tuple
        Location of the minimum color difference in the frame
    """
    with open(file_path, 'a', encoding="utf-8") as f:
        f.write("\n" + "="*50 + "\n")
        f.write("Computer Evaluation Results\n")
        f.write("="*50 + "\n")
        
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
    
    

#---------------------------------------------------------------------------------------------------------------------------------------#   
    
    
    
def log_phone_results(file_path, std, avg_diff, max_diff, max_pos, niqe, brisque, paq2piq): 
    
    """
    Logs the results of the evaluation to a text file.

    Parameters
    ----------
    file_path : str
        File path to write the results to
    std : float
        Average standard deviation of the frames in the buffer
    avg_diff : float
        Average color difference between consecutive frames in the buffer
    max_diff : float
        Maximum color difference observed between consecutive frames in the buffer
    min_diff : float
        Minimum color difference observed between consecutive frames in the buffer
    max_pos : tuple
        Location of the maximum color difference in the frame
    min_pos : tuple
        Location of the minimum color difference in the frame
    """
    with open(file_path, 'a') as f:
        f.write("\n" + "="*50 + "\n")
        f.write("Phone Evaluation Results\n")
        f.write("="*50 + "\n")
        
        table_data = [
            ["Metric", "Value"],
            ["Avg STD - Graininess", f"{std:.3f}"],
            ["Avg Color Diff ", f"{avg_diff:.3f}"],
            ["Max Color Diff", f"{max_diff:.3f} (at {max_pos})"],
            ["NIQE", f"{niqe:.3f}"],
            ["BRISQUE", f"{brisque:.3f}"],
            ["PAQ2PIQ", f"{paq2piq:.3f}"]
        ]
        
        f.write(tabulate(table_data, headers="firstrow", tablefmt='grid'))
        f.write("\n\n")
    


# TODO: add phone camera stats to the window as well 

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
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, cv2.LINE_AA)
        
    
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
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, cv2.LINE_AA)
        
    
    cv2.imshow("Phone Stats Window", stats_frame)