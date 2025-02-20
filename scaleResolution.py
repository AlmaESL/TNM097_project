import cv2

WIDTH, HEIGHT = 512, 512
FRAME_BUFFER = 10

def resize_frame(frame, width=WIDTH, height=HEIGHT):
    """
    Resize a frame to predefined dimensions 512x512

    Parameters
    ----------
    frame : numpy.array
        Input frame or image to be resized
    width : int, optional
        Width of resized frame, defaults to WIDTH
    height : int, optional
        Height of resized frame, defaults to HEIGHT

    Returns
    -------
    resized_frame : numpy.array
        Resized frame
    """
    return cv2.resize(frame, (width, height))


def store_frame(buffer, frame):
    #store a frame in the given buffer, buffers are deques 
    buffer.append(frame)
    
    #display the oldest frame in the buffer
    if len(buffer) > FRAME_BUFFER:
        buffer.popleft()
        
    return buffer

