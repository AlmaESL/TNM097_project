import cv2
import time
from scaleResolution import resize_frame, flip
from SPD import compute_spd
from FPS import calculate_fps

cap_computer = cv2.VideoCapture(0) 
cap_phone = cv2.VideoCapture(1)


SCREEN_WIDTH = 3000
SCREEN_HEIGHT = 2000
SCREEN_DIAGONAL = 14

spd = compute_spd(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_DIAGONAL)

prev_frame_time = time.time()

while True:
    
    # Read frames from capture sources 
    ret_computer, computer_frame = cap_computer.read()
    ret_phone, phone_frame = cap_phone.read()
    
    # Compute FPS
    current_frame_time = time.time()
    fps = calculate_fps(prev_frame_time, current_frame_time)
    prev_frame_time = current_frame_time
    
    
    if not ret_computer or not ret_phone:
        print("Failed to capture frames. Exiting.")
        break
    
    
    computer_frame = resize_frame(computer_frame)
    phone_frame = resize_frame(phone_frame)
    phone_frame = flip(phone_frame)
    
    
    cv2.imshow("Computer Capture", computer_frame)
    cv2.imshow("Phone Capture", phone_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Cleanup
cap_computer.release()
cap_phone.release()
cv2.destroyAllWindows()