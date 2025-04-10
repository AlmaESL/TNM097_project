import cv2

def get_available_cameras():
    
    """
    Check and return a list of available camera indices on the machine.

    This function attempts to access up to 5 camera ports (0 through 4) 
    using OpenCV's VideoCapture and returns a list of indices for those 
    cameras that are successfully opened.

    Returns
    -------
    list of int
        A list of indices for the available cameras.
    """

    available_cameras = []
    # Check for 5 cameras 
    for i in range(5):
        
        cap = cv2.VideoCapture(i)
        
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()

    return available_cameras

