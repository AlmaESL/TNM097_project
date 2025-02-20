from collections import deque

"""
This module contains a class for a rolling buffer of frames, 30
"""
class FrameBuffer:
    """rolling buffer of 30 frames"""
    
    def __init__(self, max_size=30):
        self.buffer = deque(maxlen=max_size)
    
    def add_frame(self, frame):
        """Adds a new frame to the buffer, automatically removing the oldest."""
        self.buffer.append(frame)
    
    def is_full(self):
        """Checks if the buffer has reached its maximum size."""
        return len(self.buffer) == self.buffer.maxlen

    def get_frames(self):
        """Returns all frames in the buffer."""
        return list(self.buffer)