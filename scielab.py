import numpy as np
import cv2
from scipy.ndimage import convolve
from skimage.color import rgb2xyz, xyz2lab


def rgb_to_xyz(image):
    """Convert RGB to XYZ"""
    #return cv2.cvtColor(image, cv2.COLOR_RGB2XYZ) #default scale [0,100]
    return rgb2xyz(image) #default scale [0,1] 

def xyz_to_lab(image):
    """Convert XYZ to LAB"""
    return xyz2lab(image) 



def xyz_to_opponent(XYZ):
    """
    Convert XYZ color space to opponent color space.

    Parameters
    ----------
    XYZ : array, shape (M,N,3)
        XYZ color space image.

    Returns
    -------
    O1, O2, O3 : array, each shape (M,N)
        Opponent color space images.
    """
    
    X, Y, Z = XYZ[:,:,0], XYZ[:,:,1], XYZ[:,:,2]
    
    #Matrix to convert XYZ to opponent space
    O1 = 0.279*X + 0.72*Y - 0.107*Z
    O2 = -0.449*X + 0.29*Y - 0.077*Z
    O3 = 0.086*X - 0.59*Y + 0.501*Z
    
    return O1, O2, O3

def gauss(half_width, width):
    """
    Return a Gaussian filter of specified half width and width.

    Parameters
    ----------
    half_width : int
        Half width of the Gaussian filter.
    width : int
        Width of the output filter.

    Returns
    -------
    gauss_filter : 1D array
        Gaussian filter of specified width and half width.
    """
    alpha = 2*np.sqrt(np.log(2)) / (half_width - 1)
    x = np.arange(width) - width // 2
    gauss_filter = np.exp(-alpha**2 * x**2)
    return gauss_filter / np.sum(gauss_filter)


def sum_gauss(params, width):
    """
    Compute the sum of multiple Gaussian filters.

    This function calculates a weighted sum of Gaussian filters with varying half widths
    based on the provided parameters and returns the resulting filter.

    Parameters
    ----------
    params : list of float
        A list of parameters where the first element is the width of the output filter,
        followed by pairs of half width and weight for each Gaussian filter.
    width : int
        The width of the output filter.

    Returns
    -------
    g : numpy.array
        An array representing the summed Gaussian filter of specified width.
    """

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
    # Normalize
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    xyz_frame = rgb_to_xyz(frame_rgb)
    
    # Step 2: Convert XYZ to opponent space
    O1, O2, O3 = xyz_to_opponent(xyz_frame)

    # Step 3: Apply spatial filtering to the opponent space -> this represents the human visual system
    O1_f, O2_f, O3_f = apply_spatial_filter(O1, O2, O3, spd)
    
    # filtered_xyz = np.stack((O1_f, O2_f, O3_f), axis=-1).astype(np.float32)
    # scielab_lab = xyz_to_lab(filtered_xyz)
    
    opponent_space = np.stack((O1_f, O2_f, O3_f), axis=-1)
    
    # Go back to XYZ space 
    opponent_xyz = opponent_to_xyz(opponent_space)
    
    return opponent_space
    
    # return scielab_lab




def opponent_to_xyz(opponent_matrix):
    """
    Convert an opponent color matrix to the XYZ color space.

    Parameters
    ----------
    opponent_matrix : matrix
        opponent color matrix

    Returns
    -------
    xyz_matrix : matrix
        XYZ color matrix

    Notes
    -----
    The opponent color matrix is first unpacked into its individual channels, O1, O2, and O3.
    Then, the computed inverse transformation is applied to convert the opponent matrix to XYZ.
    Finally, the XYZ channels are stacked into a single matrix.
    """
    
    # Unpack the opponent channels
    O1 = opponent_matrix[..., 0]
    O2 = opponent_matrix[..., 1]
    O3 = opponent_matrix[..., 2]
    
    # Apply the computed inverse transformation
    X = 0.627 * O1 - 1.868 * O2 - 0.153 * O3
    Y = 1.370 * O1 + 0.935 * O2 + 1.421 * O3
    Z = 1.506 * O1 + 0.436 * O2 + 2.536 * O3
    
    # Stack into an XYZ image
    xyz_matrix = np.stack((X, Y, Z), axis=-1)
    return xyz_matrix



def opponent_to_lab(opponent_matrix):
    """
    Convert an opponent color matrix to the LAB color space.

    Parameters
    ----------
    opponent_matrix : matrix
        opponent color matrix

    Returns
    -------
    lab_matrix : matrix
        LAB color matrix

    Notes
    -----
    The opponent color matrix is first converted to XYZ using the
    opponent_to_xyz function, then converted to LAB using scikit-image's
    xyz2lab function. Finally, the L*, a*, and b* channels are clipped to
    their valid ranges to prevent overflows.
    """
    xyz_matrix = opponent_to_xyz(opponent_matrix)
    lab_matrix = xyz2lab(xyz_matrix)
    
    lab_matrix[..., 0] = np.clip(lab_matrix[..., 0], 0, 100) # L* in [0,100] 
    lab_matrix[..., 1] = np.clip(lab_matrix[..., 1], -100, 100) # a* roughly in [-100,100] 
    lab_matrix[..., 2] = np.clip(lab_matrix[..., 2], -100, 100) # b* roughly in [-100,100] 
    
    return lab_matrix




def compute_color_difference(lab_frames):
    """
    Compute the color difference between consecutive frames in LAB color space using euclidean distance.

    Parameters
    ----------
    lab_frames : list of numpy.array
        List of frames in LAB color space.

    Returns
    -------
    color_diffs : list of float
        List of color differences for each consecutive frame.
    max_color_diff : float
        Maximum color difference observed between consecutive frames.
    max_diff_loc : tuple
        Location of the maximum color difference in the frame.
    diff_maps : numpy.array
        Array of color difference maps.
    """
    diff_maps = [np.linalg.norm(lab_frames[i+1] - lab_frames[i], axis=-1) 
                 for i in range(len(lab_frames) - 1)]
    diff_maps = np.stack(diff_maps, axis=0)  # Shape: (num_frames-1, H, W)

    color_diffs = [np.mean(diff_map) for diff_map in diff_maps]
    max_color_diff = np.max(diff_maps)
    frame_idx_max, h_max, w_max = np.unravel_index(np.argmax(diff_maps), diff_maps.shape)

    return color_diffs, max_color_diff, (h_max, w_max), diff_maps
