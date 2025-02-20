import cv2

def calculate_fps(prev_frame_time, current_frame_time):
    frame_time = current_frame_time - prev_frame_time
    fps = 1 / frame_time
    print(f"fps: {fps:.3f} seconds")
    return fps


def write_to_frame(frame, fps): 
     #print to frames window converted to string format
    cv2.putText(frame, 'FPS: ' + str(int(fps)), (300, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)