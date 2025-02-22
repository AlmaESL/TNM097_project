import cv2
import numpy as np
# import matlab.engine
import time


#import seems to work 
# niqe_metric = pyiqa.create_metric('niqe').cuda()


#-------------------------file imports--------------------------------#

# Import set resolution function
from scaleResolution import resize_frame, store_frame

# Import scielab computations
from scielab import scielab, opponent_to_lab, compute_color_difference

# Import fps calculations and spd calculations
from FPS import calculate_fps, write_to_frame
from computeSPD import compute_spd

# Import deque functionality
from deque import FrameBuffer

from imported_metrics import compute_niqe_avg, compute_brisque_avg, compute_paq2piq_avg
# from imported_metrics import QualityMetrics

# TODO: import stats functionality 
from logger import log_computer_results, log_phone_results, draw_stats_window


#------------------------------ToDos----------------------------------#

# TODO: Implement deque functionality and averageing -> currently under way
# TODO: Test that the brisque, niqe and paq2piq functions work 


# TODO: create and implement a stats file and functions

#---------------------------------------------------------------------#



#-------------------------Globals consts for viewing dimensions and deque size-----------------------------------#
SCREEN_WIDTH = 3000
SCREEN_HEIGHT = 2000
SCREEN_DIAGONAL = 14
# VIEWING_DISTANCE = 15.748 -> default value = 40 cm
spd = compute_spd(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_DIAGONAL)

MAX_FRAMES = 30

# Define results file
results_file_computer = "computer_results.txt"
# results_file_phone = "phone_results.txt"

#-------------------------------------------------------------------------------------------------#




#initialize camera ports 
cap_computer = cv2.VideoCapture(0) 
# cap_phone = cv2.VideoCapture(1)

#------------------------Initialize time for fpd calculations-----------------------------#
prev_frame_time = 0 
current_frame_time = 0



#---------------Initialize quality metrics object-------------------#
# quality_metrics = QualityMetrics()


#------------------------------------Frame deque buffers--------------------------------#

# Initialize buffers
computer_frame_buffer = FrameBuffer(MAX_FRAMES)
computer_opponent_buffer = FrameBuffer(MAX_FRAMES)

# phone_frame_buffer = FrameBuffer(MAX_FRAMES)
# phone_opponent_buffer = FrameBuffer(MAX_FRAMES)


# Fill buffers for both cameras  
while not computer_frame_buffer.is_full():
    
    computer_ret, computer_frame = cap_computer.read()
    
    if computer_ret: 
        
        # Resize frames 
        computer_frame = resize_frame(computer_frame)
        # print("Resized computer frame size: ", computer_frame.shape)
        
        # Compute scielab and the average std of the frames in the buffer 
        computer_frame_buffer.add_frame(computer_frame)
        computer_opponent_buffer.add_frame(scielab(computer_frame, spd))
       
    # phone_ret = cap_computer.read()

    # if phone_ret: 
        
    #     phone_frame = resize_frame(phone_frame)
    #     print("Resized phone frame size: ", phone_frame.shape)
        
    #     phone_frame_buffer.add_frame(phone_frame)
    #     phone_opponent_buffer.add_frame(scielab(phone_frame, spd))
    
    # Small delay of capture for smoother capture  
    time.sleep(0.1)


#------------------------------------Main loop--------------------------------#
while True: 
    
    computer_ret, computer_frame = cap_computer.read()
    # phone_ret, phone_frame = cap_phone.read()
    
    if not computer_ret:
        print("Failed to capture frames")
        break
    
    # Resize frames and compute scielab
    computer_frame = resize_frame(computer_frame)
    # print("frame size: ", computer_frame.shape)
    
    # Get current frame time
    current_frame_time = time.time()
    # Compute fps
    fps = calculate_fps(prev_frame_time, current_frame_time)
    # Update previous frame time to current frame time
    prev_frame_time = current_frame_time

    # #print to frames window converted to string format
    # write_to_frame(computer_frame, fps)
    
    computer_frame_buffer.add_frame(computer_frame)
    computer_opponent_matrix = scielab(computer_frame, spd)
    computer_opponent_buffer.add_frame(computer_opponent_matrix)
    
    # phone_frame = resize_frame(phone_frame)
    # phone_frame_buffer.add_frame(phone_frame)
    # phone_opponent_matrix = scielab(phone_frame, spd)
    # phone_opponent_buffer.add_frame(phone_opponent_matrix)
    
    # When buffers are full, compute the average standard deviation of the frames in the buffer - this is a graininess value 
    if computer_opponent_buffer.is_full():
        
        # std of buffer frames 
        computer_frames_array = np.stack(computer_opponent_buffer.get_frames(), axis=0)
        computer_avg_std = np.mean(np.std(computer_frames_array, axis=0))
        
        # phone_frames_array = np.stack(phone_opponent_buffer.get_frames(), axis=0)
        # phone_avg_std = np.mean(np.std(phone_frames_array, axis=0))
        
        # Euclidean distances of colors 
        computer_lab_frames = [opponent_to_lab(frame) for frame in computer_opponent_buffer.get_frames()]
        # phone_lab_frames = [opponent_to_lab(frame) for frame in phone_opponent_buffer.get_frames()]
        
        # Compute Color Differences in LAB Space
        comp_avg_diff, comp_max_diff, comp_min_diff, comp_max_pos, comp_min_pos = compute_color_difference(computer_lab_frames)
        # phone_avg_diff, phone_max_diff, phone_min_diff, phone_max_pos, phone_min_pos = compute_color_difference(phone_lab_frames)
        
        # Start computing NIQE, BRISQUE, and PaQ2PiQ asynchronously
        # quality_metrics.compute_metrics_async(computer_frame_buffer.get_frames())

        # # Retrieve the latest values (non-blocking)
        # computer_niqe, computer_brisque, computer_paq2piq = quality_metrics.get_metrics()
        
        computer_niqe = compute_niqe_avg(computer_frame_buffer.get_frames())
        computer_brisque = compute_brisque_avg(computer_frame_buffer.get_frames())
        computer_paq2piq = compute_paq2piq_avg(computer_frame_buffer.get_frames())
        
        # Print results of cielab and color differences
        # print(f"Computer Cam - Avg STD: {computer_avg_std:.3f} | Avg Color Diff: {comp_avg_diff:.3f} | Max Diff: {comp_max_diff:.3f} (at {comp_max_pos}) | Min Diff: {comp_min_diff:.3f} (at {comp_min_pos})")
        # print(f"Phone Cam - Avg STD: {phone_avg_std:.3f} | Avg Color Diff: {phone_avg_diff:.3f} | Max Diff: {phone_max_diff:.3f} (at {phone_max_pos}) | Min Diff: {phone_min_diff:.3f} (at {phone_min_pos})")

        # Log results to file
        log_computer_results(results_file_computer, computer_avg_std, comp_avg_diff, comp_max_diff, comp_min_diff, comp_max_pos, comp_min_pos, computer_niqe, computer_brisque, computer_paq2piq)
        
        # Update stats window
        stats_data = {
        "std": computer_avg_std,
        "avg_diff": comp_avg_diff,
        "max_diff": comp_max_diff,
        "min_diff": comp_min_diff,
        "max_pos": comp_max_pos,
        "min_pos": comp_min_pos,
        "niqe": computer_niqe,
        "brisque": computer_brisque,
        "paq2piq": computer_paq2piq
        }
        
        draw_stats_window(stats_data)


    # Display frames from both cameras
    cv2.imshow("Computer capture", computer_frame)
    # cv2.imshow("Phone capture", phone_frame)
        
    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#cleanup 
cap_computer.release()
# cap_phone.release()
cv2.destroyAllWindows()










#-------------------------Camera test ----------------------------------#
# cam_cap = cv2.VideoCapture(2)
# ret, frame = cam_cap.read()
    
# while True: 
#     ret, frame = cam_cap.read()
    
#     if ret: 
#         frame = resize_frame(frame)
#         cv2.imshow("Camera", frame)
#         print("frame szie:", frame.shape)
#         cv2.imwrite("test.tif", frame)
        
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     else: 
#         print("Failed to capture frame")
#         break
    


# #cleanup 
# cam_cap.release()
# cv2.destroyAllWindows()

   
