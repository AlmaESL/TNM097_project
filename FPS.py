import cv2

def calculate_fps(prev_frame_time, current_frame_time):
    """
    Calculate the time elapsed since the last frame and calculate the frames per second (FPS) accordingly.

    Parameters
    ----------
    prev_frame_time : float
        The time of the previous frame in seconds.
    current_frame_time : float
        The time of the current frame in seconds.

    Returns
    -------
    fps : float
        The frames per second since the last frame.
    """
    frame_time = current_frame_time - prev_frame_time
    fps = 1 / frame_time
    print(f"fps: {fps:.3f} seconds\n")
    return fps


def write_to_frame(frame, fps): 

    """
    Write the frames per second to the frame using OpenCV's cv2.putText.

    Parameters
    ----------
    frame : numpy.array
        The frame to write to.
    fps : float
        The frames per second to write to the frame.

    Returns
    -------
    None
    """
    # Print to frames window converted to string format
    cv2.putText(frame, 'FPS: ' + str(int(fps)), (300, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)