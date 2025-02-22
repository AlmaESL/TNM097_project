import numpy as np
import cv2
from scipy.ndimage import convolve
from scipy.spatial.distance import euclidean
from skimage.color import rgb2xyz, xyz2lab


def rgb_to_xyz(image):
    """Convert RGB to XYZ"""
    #return cv2.cvtColor(image, cv2.COLOR_RGB2XYZ) #default scale [0,100]
    return rgb2xyz(image) #default scale [0,1] 

def xyz_to_lab(image):
    """Convert XYZ to LAB"""
    return xyz2lab(image) 

def xyz_to_opponent(XYZ):
    """Convert XYZ image to opponent space"""
    X, Y, Z = XYZ[:,:,0], XYZ[:,:,1], XYZ[:,:,2]
    
    #Matrix to convert XYZ to opponent space
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

    """
    Apply spatial filtering to the opponent color space components.

    This function performs a convolution of each opponent component with a
    Gaussian kernel, ensuring the width of the kernel is odd. The Gaussian
    kernels are generated based on predefined parameters for each component.

    Parameters
    ----------
    O1 : numpy.array
        First opponent component of the image.
    O2 : numpy.array
        Second opponent component of the image.
    O3 : numpy.array
        Third opponent component of the image.
    spd : float
        Samples Per Degree of the screen, used to determine the width of
        the Gaussian kernel.

    Returns
    -------
    O1_f : numpy.array
        Filtered first opponent component.
    O2_f : numpy.array
        Filtered second opponent component.
    O3_f : numpy.array
        Filtered third opponent component.
    """

    width = int(spd / 2) * 2 - 1  # Ensure odd width 
    
    # Parameters for the Gaussian kernels - pre-defined for human visual system
    x1 = [width, 0.05, 1.00327, 0.225, 0.114416, 7.0, -0.117686]
    x2 = [width, 0.0685, 0.616725, 0.826, 0.383275]
    x3 = [width, 0.0920, 0.567885, 0.6451, 0.432115]

    k1 = sum_gauss(x1, width)
    k2 = sum_gauss(x2, width)
    k3 = sum_gauss(x3, width)

    # Convolve with reflection padding 
    O1_f = convolve(O1, k1[:, None], mode='reflect')
    O2_f = convolve(O2, k2[:, None], mode='reflect')
    O3_f = convolve(O3, k3[:, None], mode='reflect')

    return O1_f, O2_f, O3_f



def scielab(frame, spd):
    """
    Compute the S-CIELAB color space representation of an image in Lab space

    Parameters
    ----------
    frame : matrix
        Input image or frame in RGB
    spd : float, optional
        Samples Per Degree of the screen, defaults to 70

    Returns
    -------
    scielab_lab : matrix
        S-CIELAB image
    """
    
    # Step 1: Convert RGB to XYZ
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    xyz_frame = rgb_to_xyz(frame_rgb)
    
    # Step 2: Convert XYZ to opponent space
    O1, O2, O3 = xyz_to_opponent(xyz_frame)

    # Step 3: Apply spatial filtering to the opponent space -> this represents the human visual system
    O1_f, O2_f, O3_f = apply_spatial_filter(O1, O2, O3, spd)
    
    # filtered_xyz = np.stack((O1_f, O2_f, O3_f), axis=-1).astype(np.float32)
    # scielab_lab = xyz_to_lab(filtered_xyz)
    
    opponent_space = np.stack((O1_f, O2_f, O3_f), axis=-1)
    return opponent_space 
    
    # return scielab_lab


def compute_std(frames_deque): 
    """
    Compute the average standard deviation of frames in a deque

    Parameters
    ----------
    frames_deque : deque
        deque of frames

    Returns
    -------
    avg_std : matrix
        average standard deviation of frames in the deque
    """

    frames_deque = np.stack(frames_deque, axis=0)  # Shape: (30, H, W, 3)
    return np.mean(np.std(frames_deque, axis=0))



def euclidean_distance(color1, color2): 
    """
    Compute the Euclidean distance between two colors

    Parameters
    ----------
    color1 : tuple
        RGB color 1
    color2 : tuple
        RGB color 2

    Returns
    -------
    distance : float
        Euclidean distance between the two colors
    """
    return euclidean(color1, color2)

def opponent_to_lab(opponent_matrix): 

    """
    Convert opponent space to LAB

    Parameters
    ----------
    opponent_matrix : numpy.array
        Opponent space matrix

    Returns
    -------
    lab_matrix : numpy.array
        LAB matrix
    """
    
    # The inverse of the opponent transformation matrix will get us back to XYZ space
    X = (opponent_matrix[..., 0 ] + 0.107 * opponent_matrix[...,2]) / 0.279
    Y = (opponent_matrix[..., 0] - 0.72 * opponent_matrix[..., 1]) / 0.72
    Z = (opponent_matrix[..., 2] - 0.086 * opponent_matrix[..., 0]) / 0.501
    
    # Stack the XYZ channels to form the XYZ matrix
    xyz_matrix = np.stack((X,Y,Z), axis=-1)
    
    # Convert the XYZ matrix to LAB space
    lab_matrix = xyz2lab(xyz_matrix)
    
    return lab_matrix


# def compute_color_difference(lab_frames): 
#     """
#     Compute the color difference between consecutive frames in LAB color space.

#     Parameters
#     ----------
#     lab_frames : list of numpy.array
#         List of frames in LAB color space.

#     Returns
#     -------
#     avg_color_diff : float
#         Average color difference across all frames.
#     max_color_diff : float
#         Maximum color difference observed between consecutive frames.
#     min_color_diff : float
#         Minimum color difference observed between consecutive frames.
#     max_diff_loc : tuple
#         Location of the maximum color difference in the frame.
#     min_diff_loc : tuple
#         Location of the minimum color difference in the frame.
#     """
#     # Initialize the color difference maps
#     diff_maps = []
    
#     # Iterate over the frames and compute the color difference between consecutive frames
#     for i in range(len(lab_frames) - 1):
#         diff = np.linalg.norm(lab_frames[i+1] - lab_frames[i], axis=-1)
#         diff_maps.append(diff)
        
#     diff_maps = np.stack(diff_maps, axis=0)
    
#     avg_color_diff = np.mean(diff_maps)
#     max_color_diff = np.max(diff_maps)
#     min_color_diff = np.min(diff_maps)
    
#     # Retrieve the locations of min and max diffs 
#     max_diff_loc = np.unravel_index(np.argmax(diff_maps), diff_maps.shape[1:])
#     min_diff_loc = np.unravel_index(np.argmin(diff_maps), diff_maps.shape[1:])
    
#     return avg_color_diff, max_color_diff, min_color_diff, max_diff_loc, min_diff_loc


def compute_color_difference(lab_frames): 
    """
    Compute the color difference between consecutive frames in LAB color space.

    Parameters
    ----------
    lab_frames : list of numpy.array
        List of frames in LAB color space.

    Returns
    -------
    avg_color_diff : float
        Average color difference across all frames.
    max_color_diff : float
        Maximum color difference observed between consecutive frames.
    min_color_diff : float
        Minimum color difference observed between consecutive frames.
    max_diff_loc : tuple
        Location of the maximum color difference in the frame.
    min_diff_loc : tuple
        Location of the minimum color difference in the frame.
    """
    # Compute frame-to-frame color difference
    diff_maps = [np.linalg.norm(lab_frames[i+1] - lab_frames[i], axis=-1) for i in range(len(lab_frames) - 1)]
    diff_maps = np.stack(diff_maps, axis=0)  # Shape: (num_frames-1, H, W)

    avg_color_diff = np.mean(diff_maps)
    max_color_diff = np.max(diff_maps)
    min_color_diff = np.min(diff_maps)

    # Correctly find the frame index and spatial location
    frame_idx_max, h_max, w_max = np.unravel_index(np.argmax(diff_maps), diff_maps.shape)
    frame_idx_min, h_min, w_min = np.unravel_index(np.argmin(diff_maps), diff_maps.shape)

    return avg_color_diff, max_color_diff, min_color_diff, (h_max, w_max), (h_min, w_min)
