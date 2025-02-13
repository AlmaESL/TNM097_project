import cv2
import numpy as np
# import matlab.engine

#import set resolution function
from scaleResolution import set_resolution



#initialize camera ports 
cap_computer = cv2.VideoCapture(0) 
cap_phone = cv2.VideoCapture(1)

#check if camera ports are open
if not cap_computer.isOpened() and cap_phone.isOpened():
    print("Error")
    exit()


#initilialize window frames for computer cam, phone cam and stats 
computer_window = "Computer Camera"
phone_window = "Phone Camera"
stats_window = "Stats"

cv2.namedWindow(computer_window)
cv2.namedWindow(phone_window)
cv2.namedWindow(stats_window)

#main loop to run video frames 
while True: 

    #read frames from computer and phone
    ret_computer, computer_frame = cap_computer.read()
    ret_phone, phone_frame = cap_phone.read()
    
    #check that frames were captured correctly
    if not ret_computer:
        print("Failed to capture frame from computer camera.")
        break
    if not ret_phone:
        print("Failed to capture frame from phone camera (iVCam).")
        break
    
    #rescale camera resolutions
    resized_computer_frame = set_resolution(computer_frame, 400, 400)
    resized_phone_frame = set_resolution(phone_frame, 00, 400)

    #display frames
    cv2.imshow(computer_window, resized_computer_frame)
    cv2.imshow(phone_window, resized_phone_frame)

    #display stats on stats window 
    stats_frame = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.putText(stats_frame, "info text goes here!", (150, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow(stats_window, stats_frame)

    #wait for user input to exit 
    if cv2.waitKey(1) == ord('q'):
        break


#-----------------------------------------------------------------------------------------------------#
#cleanup
cap_computer.release()
cap_phone.release()
cv2.destroyAllWindows()