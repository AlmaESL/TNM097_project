import cv2
import numpy as np
# import matlab.engine
import time

import pyiqa #BRISQUE, NIQUE, PaQ2PiQ and A LOT more!


#import seems to work 
# niqe_metric = pyiqa.create_metric('niqe').cuda()




#-------------------------file imports--------------------------------#

# Import set resolution function
from scaleResolution import resize_frame, store_frame

# Import scielab computations
from scielab import scielab, opponent_to_lab, compute_color_difference

# Import fps calculations and spd calculations
from FPS import calculate_fps
from computeSPD import compute_spd

# Import deque functionality
from deque import FrameBuffer

# TODO: import NIQE, BRISQUE, PaQ2PiQ functionality
# TODO: import stats functionality 


#------------------------------ToDos----------------------------------#

# TODO: Implement deque functionality and averageing -> currently under way

# TODO: create and implement a niqe file and function 
# TODO: create and implement a brisque file and function 
# TODO: create and implement a PaQ2PiQ file and function
# TODO: create and implement a stats file and functions

#---------------------------------------------------------------------#



#-------------------------Globals consts for viewing dimensions and deque size-----------------------------------#
SCREEN_WIDTH = 3000
SCREEN_HEIGHT = 2000
SCREEN_DIAGONAL = 14
# VIEWING_DISTANCE = 15.748 -> default value = 40 cm
spd = compute_spd(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_DIAGONAL)
print("spd: ", spd)

MAX_FRAMES = 30

#-------------------------------------------------------------------------------------------------#

#initialize camera ports 
cap_computer = cv2.VideoCapture(0) 
cap_phone = cv2.VideoCapture(1)




#------------------------------------Frame deque buffers--------------------------------#

# Initialize buffers
computer_frame_buffer = FrameBuffer(MAX_FRAMES)
computer_opponent_buffer = FrameBuffer(MAX_FRAMES)

phone_frame_buffer = FrameBuffer(MAX_FRAMES)
phone_opponent_buffer = FrameBuffer(MAX_FRAMES)


# Fill buffers for both cameras  
while not computer_frame_buffer.is_full() or not phone_frame_buffer.is_full():
    
    computer_ret, computer_frame = cap_computer.read()
    
    if computer_ret: 
        
        # Resize frames 
        copmuter_frame = resize_frame(computer_frame)
        print("Resized computer frame size: ", computer_frame.shape)
        
        # Compute scielab and the average std of the frames in the buffer 
        computer_frame_buffer.add_frame(computer_frame)
        computer_opponent_buffer.add_frame(scielab(computer_frame, spd))
       
    phone_ret, phone_frame = cap_phone.read()

    if phone_ret: 
        
        phone_frame = resize_frame(phone_frame)
        print("Resized phone frame size: ", phone_frame.shape)
        
        phone_frame_buffer.add_frame(phone_frame)
        phone_opponent_buffer.add_frame(scielab(phone_frame, spd))
    
    # Small delay of capture for smoother capture  
    time.sleep(0.03)


#------------------------------------Main loop--------------------------------#
while True: 
    
    computer_ret, computer_frame = cap_computer.read()
    phone_ret, phone_frame = cap_phone.read()
    
    if not computer_ret or not phone_ret:
        print("Failed to capture frames")
        break
    
    # Resize frames and compute scielab
    computer_frame = resize_frame(computer_frame)
    computer_frame_buffer.add_frame(computer_frame)
    computer_opponent_matrix = scielab(computer_frame, spd)
    computer_opponent_buffer.add_frame(computer_opponent_matrix)
    
    phone_frame = resize_frame(phone_frame)
    phone_frame_buffer.add_frame(phone_frame)
    phone_opponent_matrix = scielab(phone_frame, spd)
    phone_opponent_buffer.add_frame(phone_opponent_matrix)
    
    # When buffers are full, compute the average standard deviation of the frames in the buffer - this is a graininess value 
    if computer_opponent_buffer.is_full() and phone_opponent_buffer.is_full():
        
        # std of buffer frames 
        computer_frames_array = np.stack(computer_opponent_buffer.get_frames(), axis=0)
        computer_avg_std = np.mean(np.std(computer_frames_array, axis=0))
        
        phone_frames_array = np.stack(phone_opponent_buffer.get_frames(), axis=0)
        phone_avg_std = np.mean(np.std(phone_frames_array, axis=0))
        
        # Euclidean distances of colors 
        computer_lab_frames = [opponent_to_lab(frame) for frame in computer_opponent_buffer.get_frames()]
        phone_lab_frames = [opponent_to_lab(frame) for frame in phone_opponent_buffer.get_frames()]
        
        # Compute Color Differences in LAB Space
        comp_avg_diff, comp_max_diff, comp_min_diff, comp_max_pos, comp_min_pos = compute_color_difference(computer_lab_frames)
        phone_avg_diff, phone_max_diff, phone_min_diff, phone_max_pos, phone_min_pos = compute_color_difference(phone_lab_frames)
        
        # Print results of cielab and color differences
        print(f"Computer Cam - Avg STD: {computer_avg_std:.3f} | Avg Color Diff: {comp_avg_diff:.3f} | Max Diff: {comp_max_diff:.3f} (at {comp_max_pos}) | Min Diff: {comp_min_diff:.3f} (at {comp_min_pos})")
        print(f"Phone Cam - Avg STD: {phone_avg_std:.3f} | Avg Color Diff: {phone_avg_diff:.3f} | Max Diff: {phone_max_diff:.3f} (at {phone_max_pos}) | Min Diff: {phone_min_diff:.3f} (at {phone_min_pos})")



    # Display frames from both cameras
    cv2.imshow("Computer capture", computer_frame)
    cv2.imshow("Phone capture", phone_frame)
        
    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#cleanup 
cap_computer.release()
cv2.destroyAllWindows()