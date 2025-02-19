import cv2
import numpy as np
# import matlab.engine
import pyiqa #BRISQUE, NIQUE, PaQ2PiQ and A LOT more!

#import seems to work 
# niqe_metric = pyiqa.create_metric('niqe').cuda()




#-------------------------file imports--------------------------------#

#import set resolution function
from scaleResolution import resize_frame, store_frame
# TODO: import scielab function 
from scielab import scielab

# TODO: import spd functionality 
from computeSPD import compute_spd

# TODO: import stats functionality 


#------------------------------ToDos----------------------------------#

# TODO: create and implement scielab file and function
# TODO: create and implement an spd file and function 
# TODO: create and implement a niqe file and function 
# TODO: create and implement a brisque file and function 
# TODO: create and implement a paq2piq file and function
# TODO: create and implement a stats file and functions

#---------------------------------------------------------------------#



#-------------------------Globals consts for viewing dimensions-----------------------------------#
SCREEN_WIDTH = 3000
SCREEN_HEIGHT = 2000
SCREEN_DIAGONAL = 14
# VIEWING_DISTANCE = 15.748 -> default value = 40 cm
spd = compute_spd(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_DIAGONAL)

#-------------------------------------------------------------------------------------------------#

#initialize camera ports 
cap_computer = cv2.VideoCapture(0) 
# cap_phone = cv2.VideoCapture(1)

ret, frame = cap_computer.read()
frame_resized = resize_frame(frame)

#save the frame as png
cv2.imwrite("frame.png", frame_resized)


cap_computer.release()

if ret:
    result = scielab(frame_resized, spd)
    print("color values: ",result)
    print("dimensions: ", result.shape)

# cv2.imwrite("result.png", result)
#check if camera ports are open
# if not cap_computer.isOpened() and cap_phone.isOpened():
#     print("Error")
#     exit()

# if not cap_computer.isOpened(): 
#     print("error")
#     exit()

#initilialize window frames for computer cam, phone cam and stats 
# computer_window = "Computer Camera"
# phone_window = "Phone Camera"
# stats_window = "Stats"

# cv2.namedWindow(computer_window)
# cv2.namedWindow(phone_window)
# cv2.namedWindow(stats_window)


#initialize frame buffers 
computer_buffer = []
#phone_buffer = []

# #-----------------------------main loop to run video frames----------------------------# 
# while True: 

#     #read frames from computer and phone
#     ret_computer, computer_frame = cap_computer.read()
#     # ret_phone, phone_frame = cap_phone.read()
    
#     #check that frames were captured correctly
#     if not ret_computer:
#         print("Failed to capture frame from computer camera")
#         break
#     # if not ret_phone:
#     #     print("Failed to capture frame from phone camera (iVCam)")
#     #     break
    
#     #resize frames 
#     # computer_frame_resized = resize_frame(computer_frame)
#     # phone_frame_resized = resize_frame(phone_frame)
    
#     # #print the frame sizes - debug 
#     # print(f"Computer frame size: {computer_frame_resized.shape}")
#     # print(f"Phone frame size: {phone_frame_resized.shape}")
    
#     # #save frame to buffer
#     # store_frame(computer_buffer, computer_frame_resized)
#     # store_frame(phone_buffer, phone_frame_resized)
    
#     #compute the scielab value matrix for every frame
#     computer_scielab = scielab(computer_frame, 70)
#     print(computer_frame)
    
#     #display frames
#     cv2.imshow(computer_window, computer_frame)
#     # cv2.imshow(phone_window, phone_frame)
    
#     #display stats on stats window 
#     stats_frame = np.zeros((300, 400, 3), dtype=np.uint8)
#     cv2.putText(stats_frame, "info text goes here!", (150, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
#     cv2.imshow(stats_window, stats_frame)

#     #wait for user input to exit 
#     if cv2.waitKey(1) == ord('q'):
#         break


# #-----------------------------------------------------------------------------------------------------#
# #cleanup
# cap_computer.release()
# # cap_phone.release()
# cv2.destroyAllWindows()