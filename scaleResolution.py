import cv2

WIDTH, HEIGHT = 400, 400


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
    frame = cv2.resize(frame, (width, height))
    
    # Remove top 40 pixels
    frame = frame[40:]
    
    return frame


def flip(frame): 
    return cv2.flip(frame, 1)



