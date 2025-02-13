import cv2

WIDTH, HEIGHT = 512, 512
FRAME_BUFFER = 10

def resize_frame(frame, width=WIDTH, height=HEIGHT):
    #resize a frame to the given dimensions 
    return cv2.resize(frame, (width, height))


def store_frame(buffer, frame):
    #store a frame in the given buffer, buffers are deques 
    buffer.append(frame)
    
    #display the oldest frame in the buffer
    if len(buffer) > FRAME_BUFFER:
        buffer.popleft()
        
    return buffer