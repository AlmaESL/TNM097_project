import cv2
import numpy as np
from collections import deque
import time
import os


#-------------------------file imports--------------------------------#
print("Loading automatic models...\n")
from scaleResolution import resize_frame, flip
from scielab import scielab, opponent_to_lab, compute_color_difference
from FPS import calculate_fps
from SPD import compute_spd
from importedMetrics import compute_metrics
from logger import (
    log_computer_results, log_phone_results, draw_stats_window_computer, 
    draw_stats_window_phone, save_color_difference_maps,
    get_computer_frames_dir, get_phone_frames_dir,
    get_computer_diff_dir
)


#-------------------------Globals consts for viewing dimensions, deque size and frame counter-----------------------------------#
SCREEN_WIDTH = 3000
SCREEN_HEIGHT = 2000
SCREEN_DIAGONAL = 14

spd = compute_spd(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_DIAGONAL)
counter = 0

MAX_FRAMES = 30  
FRAME_DELAY = 0.01


#-----------------------------Directories for results logging------------------------------------#
computer_frames_dir = get_computer_frames_dir()
phone_frames_dir = get_phone_frames_dir()
diff_dir = get_computer_diff_dir()
# phone_diff_dir = get_phone_diff_dir()

print("\nCapturing frames...\n")
#------------------------------------------------Initiliaze video captures-------------------------------------------------------#
cap_computer = cv2.VideoCapture(0) 
cap_phone = cv2.VideoCapture(1)



#------------------------Initialize time for fpd calculations-----------------------------#
prev_frame_time = time.time()


#--------------------------------------Initialize frame deques--------------------------------------#
computer_frame_buffer = deque(maxlen=MAX_FRAMES)
computer_opponent_buffer = deque(maxlen=MAX_FRAMES)
phone_frame_buffer = deque(maxlen=MAX_FRAMES)
phone_opponent_buffer = deque(maxlen=MAX_FRAMES)

# Initialize the metrics deques
computer_metrics = { 'std': deque(maxlen=MAX_FRAMES), 'niqe': deque(maxlen=MAX_FRAMES),
                     'brisque': deque(maxlen=MAX_FRAMES), 'paq2piq': deque(maxlen=MAX_FRAMES), 
                     'nima': deque(maxlen=MAX_FRAMES), 
                     'piqe': deque(maxlen=MAX_FRAMES),
                     'color_diff': deque(maxlen=MAX_FRAMES) }

phone_metrics = { 'std': deque(maxlen=MAX_FRAMES), 'niqe': deque(maxlen=MAX_FRAMES),
                  'brisque': deque(maxlen=MAX_FRAMES), 'paq2piq': deque(maxlen=MAX_FRAMES),
                  'nima': deque(maxlen=MAX_FRAMES),
                  'piqe': deque(maxlen=MAX_FRAMES),
                  'color_diff': deque(maxlen=MAX_FRAMES) }

#-----------------------------------------------------------------------------------------------------#



def compute_std(frame): 
    std = 0 
    for channel in range(3):
        std_channel = np.std(frame[channel, :, :])
        std += std_channel
    return std



#------------------------------------Main loop--------------------------------#
while True:
    
    # Count frames captured, for logging
    counter += 1
    
    # Read frames from capture souces 
    ret_computer, computer_frame = cap_computer.read()
    ret_phone, phone_frame = cap_phone.read()
    
    if not ret_computer or not ret_phone:
        print("Failed to capture frames. Exiting.")
        break
    
    # Compute FPS
    # current_frame_time = time.time()
    # fps = calculate_fps(prev_frame_time, current_frame_time)
    # prev_frame_time = current_frame_time
    
    
    computer_frame = resize_frame(computer_frame)
    phone_frame = resize_frame(phone_frame)
    phone_frame = flip(phone_frame)
    
    # Display live video feed
    # cv2.imshow("Computer Capture", computer_frame)
    # cv2.imshow("Phone Capture", phone_frame)
    
    # Save frames to designated folder
    computer_frame_path = os.path.join(computer_frames_dir, f"computer_frame_{counter}.png")
    phone_frame_path = os.path.join(phone_frames_dir, f"phone_frame_{counter}.png")
    cv2.imwrite(computer_frame_path, computer_frame)  
    cv2.imwrite(phone_frame_path, phone_frame)  
    
    # Save frames for analysis
    computer_frame_buffer.append(computer_frame)
    phone_frame_buffer.append(phone_frame)
    
    
    # Small delay for smoother processing
    # time.sleep(FRAME_DELAY)
    
    # Exit main loop when 'q' is pressed
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    
    # Exit when counter exceeds MAX_FRAMES
    if counter >= MAX_FRAMES:
        break


# Cleanup
cap_computer.release()
cap_phone.release()
cv2.destroyAllWindows()

# print("\nEvaluation loop exited\n")
print("\nComputing opponent buffers...\n")

# computer_opponent_buffer.append(computer_scielab)
# phone_opponent_buffer.append(phone_scielab)


# Convert to opponent color space and add to opponent buffers
comp_counter = 0
for computer_frame in computer_frame_buffer:
    computer_opponent_buffer.append(scielab(computer_frame, spd))
    print("Added computer frame no. ", comp_counter)
    comp_counter += 1

print("\n")

phone_counter = 0
for phone_frame in phone_frame_buffer:
    phone_opponent_buffer.append(scielab(phone_frame, spd))
    print("Added phone frame no. ", phone_counter)
    phone_counter += 1
    
# Process every batch of size MAX_FRAMES
if len(computer_opponent_buffer) == MAX_FRAMES and len(phone_opponent_buffer) == MAX_FRAMES:
    
    print("\nComputing graininess...\n")

    computer_std_per_frame = [compute_std(frame) for frame in computer_opponent_buffer]
    phone_std_per_frame = [compute_std(frame) for frame in phone_opponent_buffer]

    # Compute average std across all processed frames
    computer_avg_std = np.mean(computer_std_per_frame)
    phone_avg_std = np.mean(phone_std_per_frame)

    print("\nComputing color differences...\n")
    
    # Convert to Lab color space
    computer_lab_frames = [opponent_to_lab(frame) for frame in computer_opponent_buffer]
    phone_lab_frames = [opponent_to_lab(frame) for frame in phone_opponent_buffer]
        
    device_diff_maps = []
    device_color_diff_avgs = []
    device_max_diffs = []
    device_max_positions = []
    device_max_indices = []

    for i, (comp_lab, phone_lab) in enumerate(zip(computer_lab_frames, phone_lab_frames)):
        avg_diff, max_diff, max_pos, diff_map = compute_color_difference(comp_lab, phone_lab)
    
        device_diff_maps.append(diff_map)
        device_color_diff_avgs.append(avg_diff)
        device_max_diffs.append(max_diff)
        device_max_positions.append(max_pos)
        device_max_indices.append(i)

    device_color_diff_avg = np.mean(device_color_diff_avgs)
    print("device_color_diff_avg: ", device_color_diff_avg)
        
    device_max_diff = np.max(device_max_diffs)
    device_max_pos = device_max_positions[np.argmax(device_max_diffs)]
    max_diff_frame_index = device_max_indices[np.argmax(device_max_diffs)]
    print("max color diff at index: ", max_diff_frame_index)
        
    computer_metrics['color_diff'].append(device_color_diff_avg)
    phone_metrics['color_diff'].append(device_color_diff_avg)
    
    print("\nComputing automatic metrics...\n")

    # Compute NIQE, BRISQUE, and PaQ2PiQ metrics
    computer_quality_metrics = compute_metrics(computer_frame)
    phone_quality_metrics = compute_metrics(phone_frame)

    for metric in ['niqe', 'brisque', 'paq2piq', 'nima', 'piqe']:
        computer_metrics[metric].append(computer_quality_metrics[metric])
        phone_metrics[metric].append(phone_quality_metrics[metric])
            
    computer_avg_niqe = np.mean(computer_metrics['niqe'])
    computer_avg_brisque = np.mean(computer_metrics['brisque'])
    computer_avg_paq2piq = np.mean(computer_metrics['paq2piq'])
    computer_avg_nima = np.mean(computer_metrics['nima'])
    computer_avg_piqe = np.mean(computer_metrics['piqe'])
        
    phone_avg_niqe = np.mean(phone_metrics['niqe'])
    phone_avg_brisque = np.mean(phone_metrics['brisque'])
    phone_avg_paq2piq = np.mean(phone_metrics['paq2piq'])
    phone_avg_nima = np.mean(phone_metrics['nima'])
    phone_avg_piqe = np.mean(phone_metrics['piqe'])
    
    print("\nLogging results...\n")

    # Log results to res directory as txt files
    log_computer_results(computer_avg_std, device_color_diff_avg, device_max_diff, device_max_pos, computer_avg_niqe, computer_avg_brisque, computer_avg_paq2piq, computer_avg_nima, computer_avg_piqe)
    log_phone_results(phone_avg_std, device_color_diff_avg, device_max_diff, device_max_pos, phone_avg_niqe, phone_avg_brisque, phone_avg_paq2piq, phone_avg_nima, phone_avg_piqe)

        
    # Update stats windows
    draw_stats_window_computer({
        "std": computer_avg_std,
        "avg_diff": device_color_diff_avg,
        "max_diff": device_max_diff,
        "max_pos": device_max_pos,
        "niqe": computer_avg_niqe,
        "piqe": computer_avg_piqe,
        "brisque": computer_avg_brisque,
        "nima": computer_avg_nima,
        "paq2piq": computer_avg_paq2piq  
    })
        
    draw_stats_window_phone({
        "std": phone_avg_std,
        "avg_diff": device_color_diff_avg,
        "max_diff": device_max_diff,
        "max_pos": device_max_pos,
        "niqe": phone_avg_niqe,
        "piqe": phone_avg_piqe,
        "brisque": phone_avg_brisque,
        "nima": phone_avg_nima,
        "paq2piq": phone_avg_paq2piq
    })
        
    # Save color difference maps using the logger function
    save_color_difference_maps(device_diff_maps, diff_dir, "computer")

    print("\nBatch Evaluated\n")

    # Flush deques
    computer_frame_buffer.clear()
    computer_opponent_buffer.clear()
    phone_frame_buffer.clear()
    phone_opponent_buffer.clear()
