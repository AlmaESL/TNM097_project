import cv2
import numpy as np
# import matlab.engine
import time
from collections import deque
import os
import matplotlib.pyplot as plt


#-------------------------file imports--------------------------------#

# Import set resolution function
from scaleResolution import resize_frame, resize_phone_frame #, store_frame

# Import scielab computations
from scielab import scielab, opponent_to_lab, compute_color_difference

# Import fps calculations and spd calculations
from FPS import calculate_fps, write_to_frame
from computeSPD import compute_spd

from imported_metrics import compute_metrics

from logger import log_computer_results, log_phone_results, draw_stats_window_computer, draw_stats_window_phone

from getAvailalbeCams import get_available_cameras

#------------------------------ToDos----------------------------------#





# TODO: create and implement a stats file and functions

#---------------------------------------------------------------------#



#-------------------------Globals consts for viewing dimensions and deque size-----------------------------------#
SCREEN_WIDTH = 3000
SCREEN_HEIGHT = 2000
SCREEN_DIAGONAL = 14
# VIEWING_DISTANCE = 15.748 -> default value = 40 cm
spd = compute_spd(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_DIAGONAL)

MAX_FRAMES = 30
FRAME_SKIP = 1
counter = 0

FRAME_DELAY = 0.01

# Define results file
results_file_computer = "computer_results.txt"
results_file_phone = "phone_results.txt"
# os.makedirs(results_file_computer, exist_ok=True)
# os.makedirs(results_file_phone, exist_ok=True)

# Save captured frames
computer_frames_dir = "captured_frames/computer"
phone_frames_dir = "captured_frames/phone"
os.makedirs(computer_frames_dir, exist_ok=True)
os.makedirs(phone_frames_dir, exist_ok=True)

computer_diff_dir = "color_differences/computer"
phone_diff_dir = "color_differences/phone"
os.makedirs(computer_diff_dir, exist_ok=True)
os.makedirs(phone_diff_dir, exist_ok=True)


#-------------------------------------------------------------------------------------------------#




#initialize camera ports 
cap_computer = cv2.VideoCapture(0) 
cap_phone = cv2.VideoCapture(1)

cap_computer.set(cv2.CAP_PROP_FPS, 30)
cap_phone.set(cv2.CAP_PROP_FPS, 30)

#------------------------Initialize time for fpd calculations-----------------------------#
prev_frame_time = 0 
current_frame_time = 0



#------------------------------------Frame deque buffers--------------------------------#

# Initialize buffers
computer_frame_buffer = deque(maxlen=MAX_FRAMES)
computer_opponent_buffer = deque(maxlen=MAX_FRAMES)
phone_frame_buffer = deque(maxlen=MAX_FRAMES)
phone_opponent_buffer = deque(maxlen=MAX_FRAMES)

# Rolling buffers for computed metric values, popping and pushing done automatically 
computer_metrics = {
    'std': deque(maxlen=MAX_FRAMES),
    'niqe': deque(maxlen=MAX_FRAMES),
    'brisque': deque(maxlen=MAX_FRAMES),
    'paq2piq': deque(maxlen=MAX_FRAMES),
    'color_diff': deque(maxlen=MAX_FRAMES)
}
phone_metrics = {
    'std': deque(maxlen=MAX_FRAMES),
    'niqe': deque(maxlen=MAX_FRAMES),
    'brisque': deque(maxlen=MAX_FRAMES),
    'paq2piq': deque(maxlen=MAX_FRAMES),
    'color_diff': deque(maxlen=MAX_FRAMES)
}


print("Filling deques...\n")

prev_frame_time = time.time()


#------------------------------------Main loop--------------------------------#
while  len(computer_opponent_buffer) < MAX_FRAMES and len(phone_opponent_buffer) < MAX_FRAMES: 
    counter += 1
    
    computer_ret, computer_frame = cap_computer.read()
    phone_ret, phone_frame = cap_phone.read()
    
    if not computer_ret or not phone_ret:
        print("Failed to capture frames")
        break
    
    # Only use every 3rd frame
    if counter % FRAME_SKIP != 0:
        continue
    
    # Resize frames and compute scielab
    computer_frame = resize_frame(computer_frame)
    phone_frame = resize_frame(phone_frame)
    
    # Save frames to the designated folder
    computer_frame_path = os.path.join(computer_frames_dir, f"computer_frame_{counter}.png")
    phone_frame_path = os.path.join(phone_frames_dir, f"phone_frame_{counter}.png")
    cv2.imwrite(computer_frame_path, computer_frame)  
    cv2.imwrite(phone_frame_path, phone_frame)  
    
    # Get current frame time
    current_frame_time = time.time()
    # Compute fps
    fps = calculate_fps(prev_frame_time, current_frame_time)
    # Update previous frame time to current frame time
    prev_frame_time = current_frame_time

    #print to frames window converted to string format
    # write_to_frame(computer_frame, fps)
    # write_to_frame(phone_frame, fps)
    
    # Store frames and compute opponent color space transformation
    computer_frame_buffer.append(computer_frame)
    phone_frame_buffer.append(phone_frame)
    
    computer_scielab = scielab(computer_frame, spd)
    phone_scielab = scielab(phone_frame, spd)

    computer_opponent_buffer.append(computer_scielab)
    phone_opponent_buffer.append(phone_scielab)
    
    # When buffers are full, compute the average standard deviation of the frames in the buffer - this is a graininess value 
    if len(computer_opponent_buffer) == MAX_FRAMES and len(phone_opponent_buffer) == MAX_FRAMES:
        
        # Compute standard deviation (graininess)
        new_computer_std = np.std(np.stack(computer_opponent_buffer), axis=0)
        new_phone_std = np.std(np.stack(phone_opponent_buffer), axis=0)

        computer_metrics['std'].append(new_computer_std)
        phone_metrics['std'].append(new_phone_std)

        # Convert opponent color space to LAB
        computer_lab_frames = [opponent_to_lab(frame) for frame in computer_opponent_buffer]
        phone_lab_frames = [opponent_to_lab(frame) for frame in phone_opponent_buffer]

        # Compute color difference in LAB space
        comp_avg_diff, comp_max_diff, comp_max_pos, comp_diff_maps = compute_color_difference(computer_lab_frames)
        phone_avg_diff, phone_max_diff, phone_max_pos, phone_diff_maps = compute_color_difference(phone_lab_frames)
        
        print("\nComputing...\n")
        # Save color difference maps as images with color bars
        for i, diff_map in enumerate(comp_diff_maps):
            plt.figure(figsize=(6, 6))
            plt.imshow(diff_map, cmap='viridis')  # Use a perceptually uniform colormap
            plt.colorbar(label='Color Difference')
            plt.axis('off')  # Hide axes

            # Save the figure
            save_path = os.path.join(computer_diff_dir, f"computer_diff_{i}.png")
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
            plt.close()  # Close figure to free memory

        for i, diff_map in enumerate(phone_diff_maps):
            plt.figure(figsize=(6, 6))
            plt.imshow(diff_map, cmap='viridis')
            plt.colorbar(label='Color Difference')
            plt.axis('off')

            # Save the figure
            save_path = os.path.join(phone_diff_dir, f"phone_diff_{i}.png")
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
            plt.close()
        
        
        computer_metrics['color_diff'].append(comp_avg_diff)
        phone_metrics['color_diff'].append(phone_avg_diff)

        # Compute NIQE, BRISQUE, and PaQ2PiQ metricsq
        computer_quality_metrics = compute_metrics(computer_frame)
        phone_quality_metrics = compute_metrics(phone_frame)

        for metric in ['niqe', 'brisque', 'paq2piq']:
            computer_metrics[metric].append(computer_quality_metrics[metric])
            phone_metrics[metric].append(phone_quality_metrics[metric])

        # Compute moving averages
        computer_avg_std = np.mean(computer_metrics['std'])
        phone_avg_std = np.mean(phone_metrics['std'])
        
        avg_computer_color_diff = np.mean(computer_metrics['color_diff'])
        avg_phone_color_diff = np.mean(phone_metrics['color_diff'])
        
        computer_avg_niqe = np.mean(computer_metrics['niqe'])
        computer_avg_brisque = np.mean(computer_metrics['brisque'])
        computer_avg_paq2piq = np.mean(computer_metrics['paq2piq'])
        
        phone_avg_niqe = np.mean(phone_metrics['niqe'])
        phone_avg_brisque = np.mean(phone_metrics['brisque'])
        phone_avg_paq2piq = np.mean(phone_metrics['paq2piq'])

        # Log results to txt files
        log_computer_results(results_file_computer, computer_avg_std, avg_computer_color_diff, comp_max_diff, comp_max_pos, computer_avg_niqe, computer_avg_brisque, computer_avg_paq2piq)
        log_phone_results(results_file_phone, phone_avg_std, avg_phone_color_diff, phone_max_diff, phone_max_pos, phone_avg_niqe, phone_avg_brisque, phone_avg_paq2piq)

        # # Update stats windows
        # draw_stats_window_computer({
        #     "std": computer_avg_std,
        #     "avg_diff": avg_computer_color_diff,
        #     "max_diff": comp_max_diff,
        #     "max_pos": comp_max_pos,
        #     "niqe": computer_avg_niqe,
        #     "brisque": computer_avg_brisque,
        #     "paq2piq": computer_avg_paq2piq,
        #     "fps: ": fps
        # })
        # draw_stats_window_phone({
        #     "std": phone_avg_std,
        #     "avg_diff": avg_phone_color_diff,
        #     "max_diff": phone_max_diff,
        #     "max_pos": phone_max_pos,
        #     "niqe": phone_avg_niqe,
        #     "brisque": phone_avg_brisque,
        #     "paq2piq": phone_avg_paq2piq,
        #     "fps: ": fps
        # })
        
     
    # Display frames from both cameras
    cv2.imshow("Computer capture", computer_frame)
    # cv2.imwrite("comp_frame.png", computer_frame) #debugging
    cv2.imshow("Phone capture", phone_frame)
    # cv2.imwrite("phone_frame.png", phone_frame) #debugging 
     
    # Capture delay for smoother capture    
    time.sleep(FRAME_DELAY)   
     
    # # Exit loop when 'q' is pressed
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break


# Cleanup 
cap_computer.release()
cap_phone.release()
cv2.destroyAllWindows()

print("\nEvaluation complete\n")








# # #-----------------------Check available cameras----------------------------#

# # # cams = get_available_cameras()
# # # print("number of available cameras: ", len(cams))





# # #-------------------------Camera test ----------------------------------#
# # # cam_cap = cv2.VideoCapture(1)
# # # # cam_cap = cv2.VideoCapture(3) #3 iriun webcam
# # # ret, frame = cam_cap.read()
    
# # # while True: 
# # #     ret, frame = cam_cap.read()
    
# # #     if ret: 
# # #         frame = resize_phone_frame(frame)
# # #         cv2.imshow("Camera", frame)
# # #         # print("frame szie:", frame.shape)
# # #         cv2.imwrite("test1.tif", frame)
        
# # #         if cv2.waitKey(1) & 0xFF == ord('q'):
# # #             break
# # #     else: 
# # #         print("Failed to capture frame")
# # #         break
    


# # # #cleanup 
# # # cam_cap.release()
# # # cv2.destroyAllWindows()




# # import cv2
# # import numpy as np
# # import time

# # # Import functions
# # from scaleResolution import resize_frame
# # from scielab import scielab, opponent_to_lab, compute_color_difference
# # from FPS import calculate_fps
# # from computeSPD import compute_spd
# # from deque import FrameBuffer
# # from imported_metrics import compute_niqe_avg, compute_brisque_avg, compute_paq2piq_avg
# # from logger import log_computer_results, log_phone_results

# # # Constants
# # SCREEN_WIDTH = 3000
# # SCREEN_HEIGHT = 2000
# # SCREEN_DIAGONAL = 14
# # spd = compute_spd(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_DIAGONAL)
# # MAX_FRAMES = 30

# # # Initialize cameras
# # cap_computer = cv2.VideoCapture(0)
# # cap_phone = cv2.VideoCapture(1)

# # # Initialize buffers
# # computer_frame_buffer = FrameBuffer(MAX_FRAMES)
# # computer_opponent_buffer = FrameBuffer(MAX_FRAMES)
# # phone_frame_buffer = FrameBuffer(MAX_FRAMES)
# # phone_opponent_buffer = FrameBuffer(MAX_FRAMES)

# # # Rolling buffers for computed metric values
# # computer_metrics = {'std': [], 'niqe': [], 'brisque': [], 'paq2piq': [], 'color_diff': []}
# # phone_metrics = {'std': [], 'niqe': [], 'brisque': [], 'paq2piq': [], 'color_diff': []}

# # # Fill buffers initially
# # while not computer_frame_buffer.is_full() and not phone_frame_buffer.is_full():
# #     computer_ret, computer_frame = cap_computer.read()
# #     phone_ret, phone_frame = cap_phone.read()

# #     if computer_ret:
# #         computer_frame = resize_frame(computer_frame)
# #         computer_frame_buffer.add_frame(computer_frame)
# #         computer_opponent_buffer.add_frame(scielab(computer_frame, spd))
    
# #     if phone_ret:
# #         phone_frame = resize_frame(phone_frame)
# #         phone_frame_buffer.add_frame(phone_frame)
# #         phone_opponent_buffer.add_frame(scielab(phone_frame, spd))
    
# #     time.sleep(0.1)

# # prev_frame_time = time.time()

# # # Main loop
# # while True:
# #     computer_ret, computer_frame = cap_computer.read()
# #     phone_ret, phone_frame = cap_phone.read()

# #     if not computer_ret or not phone_ret:
# #         print("Failed to capture frames")
# #         break

# #     computer_frame = resize_frame(computer_frame)
# #     phone_frame = resize_frame(phone_frame)
    
# #     # Compute FPS
# #     current_frame_time = time.time()
# #     fps = calculate_fps(prev_frame_time, current_frame_time)
# #     prev_frame_time = current_frame_time

# #     # Update buffers
# #     computer_frame_buffer.add_frame(computer_frame)
# #     new_computer_scielab = scielab(computer_frame, spd)
# #     computer_opponent_buffer.add_frame(new_computer_scielab)
    
# #     phone_frame_buffer.add_frame(phone_frame)
# #     new_phone_scielab = scielab(phone_frame, spd)
# #     phone_opponent_buffer.add_frame(new_phone_scielab)

# #     if computer_opponent_buffer.is_full() and phone_opponent_buffer.is_full():
# #         # Compute std deviation for the new frame
# #         new_computer_std = np.std(new_computer_scielab)
# #         new_phone_std = np.std(new_phone_scielab)
        
# #         if len(computer_metrics['std']) == MAX_FRAMES:
# #             computer_metrics['std'].pop(0)
# #         computer_metrics['std'].append(new_computer_std)
# #         avg_computer_std = np.mean(computer_metrics['std'])
        
# #         if len(phone_metrics['std']) == MAX_FRAMES:
# #             phone_metrics['std'].pop(0)
# #         phone_metrics['std'].append(new_phone_std)
# #         avg_phone_std = np.mean(phone_metrics['std'])
        
# #         # Compute color difference
# #         computer_lab_frames = [opponent_to_lab(frame) for frame in computer_opponent_buffer.get_frames()]
# #         phone_lab_frames = [opponent_to_lab(frame) for frame in phone_opponent_buffer.get_frames()]
# #         comp_avg_diff, comp_max_diff, comp_max_diff_pos = compute_color_difference(computer_lab_frames)
# #         phone_avg_diff, phone_max_diff, phone_max_diff_pos = compute_color_difference(phone_lab_frames)
        
# #         if len(computer_metrics['color_diff']) == MAX_FRAMES:
# #             computer_metrics['color_diff'].pop(0)
# #         computer_metrics['color_diff'].append(comp_avg_diff)
# #         avg_computer_color_diff = np.mean(computer_metrics['color_diff'])
        
# #         if len(phone_metrics['color_diff']) == MAX_FRAMES:
# #             phone_metrics['color_diff'].pop(0)
# #         phone_metrics['color_diff'].append(phone_avg_diff)
# #         avg_phone_color_diff = np.mean(phone_metrics['color_diff'])
        
# #         # Compute quality metrics for the new frame
# #         new_computer_niqe = compute_niqe_avg([computer_frame])
# #         new_computer_brisque = compute_brisque_avg([computer_frame])
# #         new_computer_paq2piq = compute_paq2piq_avg([computer_frame])

# #         new_phone_niqe = compute_niqe_avg([phone_frame])
# #         new_phone_brisque = compute_brisque_avg([phone_frame])
# #         new_phone_paq2piq = compute_paq2piq_avg([phone_frame])
        
# #         log_computer_results("computer_results.txt", avg_computer_std, avg_computer_color_diff, comp_max_diff, comp_max_diff_pos, new_computer_niqe, new_computer_brisque, new_computer_paq2piq)
# #         log_phone_results("phone_results.txt", avg_phone_std, avg_phone_color_diff, phone_max_diff, phone_max_diff_pos, new_phone_niqe, new_phone_brisque, new_phone_paq2piq)

# #     cv2.imshow("Computer capture", computer_frame)
# #     cv2.imshow("Phone capture", phone_frame)
    
# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         break

# # cap_computer.release()
# # cap_phone.release()
# # cv2.destroyAllWindows()
# # print("Evaluation complete")




# import cv2
# import numpy as np
# import torch
# import time
# import pyiqa
# from collections import deque


# from scaleResolution import resize_frame
# from scielab import scielab, opponent_to_lab, compute_color_difference
# from FPS import calculate_fps, write_to_frame
# from computeSPD import compute_spd
# from imported_metrics import compute_niqe, compute_brisque, compute_paq2piq
# from logger import log_computer_results, log_phone_results, draw_stats_window_computer, draw_stats_window_phone

# # Screen parameters
# SCREEN_WIDTH = 3000
# SCREEN_HEIGHT = 2000
# SCREEN_DIAGONAL = 14
# spd = compute_spd(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_DIAGONAL)

# MAX_FRAMES = 30
# FRAME_SKIP = 3
# FRAME_DELAY = 0.5

# # Initialize video capture
# cap_computer = cv2.VideoCapture(0)
# cap_phone = cv2.VideoCapture(1)

# # Set FPS capture rate
# cap_computer.set(cv2.CAP_PROP_FPS, 30)
# cap_phone.set(cv2.CAP_PROP_FPS, 30)

# # Initialize frame buffers
# computer_frame_buffer = deque(maxlen=MAX_FRAMES)
# computer_opponent_buffer = deque(maxlen=MAX_FRAMES)
# phone_frame_buffer = deque(maxlen=MAX_FRAMES)
# phone_opponent_buffer = deque(maxlen=MAX_FRAMES)

# # Rolling buffers for computed metric values (deque handles max length automatically)
# computer_metrics = {
#     'std': deque(maxlen=MAX_FRAMES),
#     'niqe': deque(maxlen=MAX_FRAMES),
#     'brisque': deque(maxlen=MAX_FRAMES),
#     'paq2piq': deque(maxlen=MAX_FRAMES),
#     'color_diff': deque(maxlen=MAX_FRAMES)
# }
# phone_metrics = {
#     'std': deque(maxlen=MAX_FRAMES),
#     'niqe': deque(maxlen=MAX_FRAMES),
#     'brisque': deque(maxlen=MAX_FRAMES),
#     'paq2piq': deque(maxlen=MAX_FRAMES),
#     'color_diff': deque(maxlen=MAX_FRAMES)
# }

# # Preload quality metric models
# niqe_metric = pyiqa.create_metric('niqe').to('cpu')
# paq2piq_metric = pyiqa.create_metric('paq2piq').to('cpu')

# def preprocess_frame(frame):
#     frame = frame.astype(np.float32) / 255.0
#     frame = np.transpose(frame, (2, 0, 1))
#     return torch.tensor(frame).unsqueeze(0)

# def compute_metrics(frame):
#     """Compute all quality metrics for a single frame"""
#     frame_tensor = preprocess_frame(frame)
#     brisque_metric = pyiqa.create_metric('brisque').to('cpu')
#     return {
#         'niqe': niqe_metric(frame_tensor).item(),
#         'brisque': brisque_metric(frame_tensor).item(),
#         'paq2piq': paq2piq_metric(frame_tensor).item()
#     }

# # Main loop
# counter = 0
# prev_frame_time = time.time()

# while True:
#     counter += 1
#     computer_ret, computer_frame = cap_computer.read()
#     phone_ret, phone_frame = cap_phone.read()

#     if not computer_ret or not phone_ret:
#         print("Failed to capture frames")
#         break

#     if counter % FRAME_SKIP != 0:
#         continue

#     # Resize frames
#     computer_frame = resize_frame(computer_frame)
#     phone_frame = resize_frame(phone_frame)

#     # Compute FPS
#     current_frame_time = time.time()
#     fps = calculate_fps(prev_frame_time, current_frame_time)
#     prev_frame_time = current_frame_time

#     # Display FPS on frames
#     write_to_frame(computer_frame, fps)
#     write_to_frame(phone_frame, fps)

#     # Store frames and compute opponent color space transformation
#     computer_frame_buffer.append(computer_frame)
#     phone_frame_buffer.append(phone_frame)
    
#     computer_scielab = scielab(computer_frame, spd)
#     phone_scielab = scielab(phone_frame, spd)

#     computer_opponent_buffer.append(computer_scielab)
#     phone_opponent_buffer.append(phone_scielab)

#     if len(computer_opponent_buffer) == MAX_FRAMES and len(phone_opponent_buffer) == MAX_FRAMES:
#         # Compute standard deviation (graininess)
#         new_computer_std = np.std(np.stack(computer_opponent_buffer), axis=0)
#         new_phone_std = np.std(np.stack(phone_opponent_buffer), axis=0)

#         computer_metrics['std'].append(new_computer_std)
#         phone_metrics['std'].append(new_phone_std)

#         # Convert opponent color space to LAB
#         computer_lab_frames = [opponent_to_lab(frame) for frame in computer_opponent_buffer]
#         phone_lab_frames = [opponent_to_lab(frame) for frame in phone_opponent_buffer]

#         # Compute color difference in LAB space
#         comp_avg_diff, comp_max_diff, comp_max_pos = compute_color_difference(computer_lab_frames)
#         phone_avg_diff, phone_max_diff, phone_max_pos = compute_color_difference(phone_lab_frames)

#         computer_metrics['color_diff'].append(comp_avg_diff)
#         phone_metrics['color_diff'].append(phone_avg_diff)

#         # Compute NIQE, BRISQUE, and PaQ2PiQ metrics
#         computer_quality_metrics = compute_metrics(computer_frame)
#         phone_quality_metrics = compute_metrics(phone_frame)

#         for metric in ['niqe', 'brisque', 'paq2piq']:
#             computer_metrics[metric].append(computer_quality_metrics[metric])
#             phone_metrics[metric].append(phone_quality_metrics[metric])

#         # Compute moving averages
#         computer_avg_std = np.mean(computer_metrics['std'])
#         phone_avg_std = np.mean(phone_metrics['std'])
#         avg_computer_color_diff = np.mean(computer_metrics['color_diff'])
#         avg_phone_color_diff = np.mean(phone_metrics['color_diff'])
#         computer_avg_niqe = np.mean(computer_metrics['niqe'])
#         computer_avg_brisque = np.mean(computer_metrics['brisque'])
#         computer_avg_paq2piq = np.mean(computer_metrics['paq2piq'])
#         phone_avg_niqe = np.mean(phone_metrics['niqe'])
#         phone_avg_brisque = np.mean(phone_metrics['brisque'])
#         phone_avg_paq2piq = np.mean(phone_metrics['paq2piq'])

#         # Log results
#         log_computer_results("computer_results.txt", computer_avg_std, avg_computer_color_diff, comp_max_diff, comp_max_pos, computer_avg_niqe, computer_avg_brisque, computer_avg_paq2piq)
#         log_phone_results("phone_results.txt", phone_avg_std, avg_phone_color_diff, phone_max_diff, phone_max_pos, phone_avg_niqe, phone_avg_brisque, phone_avg_paq2piq)

#         # Update UI stats
#         draw_stats_window_computer({
#             "std": computer_avg_std,
#             "avg_diff": comp_avg_diff,
#             "max_diff": comp_max_diff,
#             "max_pos": comp_max_pos,
#             "niqe": computer_avg_niqe,
#             "brisque": computer_avg_brisque,
#             "paq2piq": computer_avg_paq2piq
#         })
#         draw_stats_window_phone({
#             "std": phone_avg_std,
#             "avg_diff": phone_avg_diff,
#             "max_diff": phone_max_diff,
#             "max_pos": phone_max_pos,
#             "niqe": phone_avg_niqe,
#             "brisque": phone_avg_brisque,
#             "paq2piq": phone_avg_paq2piq
#         })

#     # Display video frames
#     cv2.imshow("Computer capture", computer_frame)
#     cv2.imshow("Phone capture", phone_frame)

#     time.sleep(FRAME_DELAY)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Cleanup
# cap_computer.release()
# cap_phone.release()
# cv2.destroyAllWindows()
