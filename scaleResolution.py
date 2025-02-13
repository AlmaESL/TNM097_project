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



#image resizing in cv2 
"""
import cv2

# Load the image
image = cv2.imread('image.jpg')

# Get the original dimensions
(h, w) = image.shape[:2]

# Desired width
new_width = 800

# Calculate the aspect ratio
aspect_ratio = h / w
new_height = int(new_width * aspect_ratio)

# Resize the image
resized_image = cv2.resize(image, (new_width, new_height))
"""