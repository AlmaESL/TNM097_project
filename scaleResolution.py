import cv2

WIDTH, HEIGHT = 512, 512
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
    return cv2.resize(frame, (width, height))



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
    # return remove_margins(frame)


# Function to remove the 64 pixel margins at top and bottom 
def remove_margins(frame):
    """
    Remove the 64 pixel margins at top and bottom of the given frame.

    Parameters
    ----------
    frame : numpy.array
        Input frame or image to be resized

    Returns
    -------
    resized_frame : numpy.array
        Frame with margins removed
    """
    return frame[64:-64]


