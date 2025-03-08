import cv2

WIDTH, HEIGHT = 400, 400
FRAME_BUFFER = 10
PHONE_HEIGHT = 640

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


def resize_phone_frame(frame, width=WIDTH, height=PHONE_HEIGHT): 
    """
    Resize a frame to predefined dimensions 512x640

    Parameters
    ----------
    frame : numpy.array
        Input frame or image to be resized
    width : int, optional
        Width of resized frame, defaults to WIDTH
    height : int, optional
        Height of resized frame, defaults to PHONE_HEIGHT

    Returns
    -------
    resized_frame : numpy.array
        Resized frame
    """
    
    return cv2.resize(frame, (width, height))


