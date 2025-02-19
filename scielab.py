import numpy as np
import cv2
from scipy.ndimage import convolve
from skimage.color import rgb2xyz, xyz2lab

def rgb_to_xyz(image):
    """Convert RGB to XYZ"""
    return cv2.cvtColor(image, cv2.COLOR_RGB2XYZ)

def xyz_to_lab(image):
    """Convert XYZ to LAB"""
    return xyz2lab(image)

def xyz_to_opponent(XYZ):
    """Convert XYZ image to opponent space"""
    X, Y, Z = XYZ[:,:,0], XYZ[:,:,1], XYZ[:,:,2]
    O1 = 0.279*X + 0.72*Y - 0.107*Z
    O2 = -0.449*X + 0.29*Y - 0.077*Z
    O3 = 0.086*X - 0.59*Y + 0.501*Z
    return O1, O2, O3

def gauss(half_width, width):
    """Generate a 1D Gaussian kernel"""
    alpha = 2*np.sqrt(np.log(2)) / (half_width - 1)
    x = np.arange(width) - width // 2
    gauss_filter = np.exp(-alpha**2 * x**2)
    return gauss_filter / np.sum(gauss_filter)

def sum_gauss(params, width):
    """Sum weighted Gaussians"""
    num_G = (len(params) - 1) // 2
    g = np.zeros(width)
    
    for i in range(num_G):
        half_width = params[2*i+1]
        weight = params[2*i+2]
        g += weight * gauss(half_width, width)
    
    return g

def apply_spatial_filter(O1, O2, O3, spd):
    """Apply spatial filtering to opponent channels"""
    width = int(spd / 2) * 2 - 1  # Ensure odd width
    
    x1 = [width, 0.05, 1.00327, 0.225, 0.114416, 7.0, -0.117686]
    x2 = [width, 0.0685, 0.616725, 0.826, 0.383275]
    x3 = [width, 0.0920, 0.567885, 0.6451, 0.432115]

    k1 = sum_gauss(x1, width)
    k2 = sum_gauss(x2, width)
    k3 = sum_gauss(x3, width)

    O1_f = convolve(O1, k1[:, None], mode='reflect')
    O2_f = convolve(O2, k2[:, None], mode='reflect')
    O3_f = convolve(O3, k3[:, None], mode='reflect')

    return O1_f, O2_f, O3_f

def scielab(frame, spd=70):
    """Compute SCIELAB for an image using given SPD"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    xyz_frame = rgb_to_xyz(frame_rgb)
    O1, O2, O3 = xyz_to_opponent(xyz_frame)

    O1_f, O2_f, O3_f = apply_spatial_filter(O1, O2, O3, spd)
    
    filtered_xyz = np.stack((O1_f, O2_f, O3_f), axis=-1).astype(np.float32)
    scielab_lab = xyz_to_lab(filtered_xyz)
    
    return scielab_lab
