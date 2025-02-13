import cv2

def set_resolution(frame, width, height): 
    
    # width = int(frame.shape[1] * scale)
    # height = int(frame.shape[0] * scale)
    
    dimensions = (width, height)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)
    