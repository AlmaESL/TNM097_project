import numpy as np
import cv2
from scipy.ndimage import convolve
from skimage.color import rgb2xyz, xyz2lab

OPP_MAT = np.array([[ 0.279,  0.722,  -0.107],
    [-0.449,  0.326,  0.077],
    [ 0.080, -0.59,   0.501]])

def rgb_to_xyz(image):
    """Convert RGB to XYZ"""
    #return cv2.cvtColor(image, cv2.COLOR_RGB2XYZ) #default scale [0,100]
    return rgb2xyz(image) #default scale [0,1] 

def xyz_to_lab(image):
    """Convert XYZ to LAB"""
    
    # convert to LAB
    image = xyz2lab(image)
    
    # # Make sure lab values are 0,1
    # image[..., 0] = np.clip(image[..., 0], 0, 100) # L* in [0,100] 
    # image[..., 1] = np.clip(image[..., 1], -100, 100) # a* roughly in [-100,100] 
    # image[..., 2] = np.clip(image[..., 2], -100, 100) # b* roughly in [-100,100]
    
    return image



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
    
    # print("o1: ", O1)
    # print("o2: ", O2)
    # print("o3: ", O3)
    
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
    
    x1[1::2] = [x * spd for x in x1[1::2]]
    x2[1::2] = [x * spd for x in x2[1::2]]
    x3[1::2] = [x * spd for x in x3[1::2]]

    k1 = sum_gauss(x1, width).astype(np.float64)
    k2 = sum_gauss(x2, width).astype(np.float64)
    k3 = sum_gauss(x3, width).astype(np.float64)


    # Convolve with reflection padding 
    O1_f = convolve(O1, k1[:, None], mode='reflect')
    O2_f = convolve(O2, k2[:, None], mode='reflect')
    O3_f = convolve(O3, k3[:, None], mode='reflect')

    return O1_f, O2_f, O3_f

7



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
    # print("xyz: ", xyz_frame)
    # print("xyz size: ", xyz_frame.shape)
    
    # Step 2: Convert XYZ to opponent space
    O1, O2, O3 = xyz_to_opponent(xyz_frame)
    # print("opponent without spatials: ", np.stack((O1, O2, O3), axis=-1))
    # print("\n\no1: ", O1)
    # print("o2: ", O2)
    # print("o3: ", O3)
    # print("size: ", np.stack((O1, O2, O3), axis=-1).shape)
    
    
    # Step 3: Apply spatial filtering to the opponent space -> this represents the human visual system
    O1_f, O2_f, O3_f = apply_spatial_filter(O1, O2, O3, spd)
    # print("o1 filtered: ", O1_f)
    # print("o2 filtered: ", O2_f)
    # print("o3 filtered: ", O3_f)
    
    # make matrix of o1_f, o2_f and o3_f
    # opponent_space = np.dstack((O1_f, O2_f, O3_f))
    opponent_space = np.array([O1_f, O2_f, O3_f])
    print("opponent matrix: ", opponent_space)
    
    return opponent_space





def opponent_to_xyz(frame):
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
    print("frame size:", frame.shape)
    # Unpack the opponent channels
    O1 = frame[0, ...]
    O2 = frame[1, ...]
    O3 = frame[2, ...]
    
    # Apply the computed inverse transformation
    X = 0.610 * O1 - 1.819 * O2 - 0.149 * O3
    Y = 1.416 * O1 + 0.798 * O2 + 0.425 * O3
    Z = 1.772 * O1 + 0.628 * O2 + 2.471 * O3
    
    # print("x_new: ", X)
    # print("y_new: ", Y)
    # print("z_new: ", Z)   
 
    # Stack into an XYZ image
    xyz_matrix = np.stack((X, Y, Z), axis=-1)
    return xyz_matrix



def opponent_to_lab(frame):
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
    # print("frame")
    xyz_matrix = opponent_to_xyz(frame)
    lab_matrix = xyz2lab(xyz_matrix)
    # print("lab: ", lab_matrix)
    
    return lab_matrix



def compute_color_difference(lab_frame1, lab_frame2):
    """
    Compute the color difference between two frames in LAB color space using Euclidean distance.

    Parameters
    ----------
    lab_frame1 : numpy.array
        First frame in LAB color space.
    lab_frame2 : numpy.array
        Second frame in LAB color space.

    Returns
    -------
    mean_color_diff : float
        Mean color difference across the image.
    max_color_diff : float
        Maximum color difference observed.
    max_diff_loc : tuple
        Location (row, col) of the maximum color difference.
    diff_map : numpy.array
        2D array of pixel-wise color differences.
    """
    
    print("lab1: ", lab_frame1.shape)
    print("lab2: ", lab_frame2.shape)
    
    
    # Ensure frames are in (H, W, 3) format
    if lab_frame1.shape[0] != 3:
        lab_frame1 = np.transpose(lab_frame1, (2, 1, 0))
        print("lab1 transposed: ", lab_frame1.shape)
    if lab_frame2.shape[0] != 3:
        lab_frame2 = np.transpose(lab_frame2, (2, 1, 0))
        print("lab2 transposed: ", lab_frame2.shape)
    
    max = 0
    # Compute Euclidean distance at each pixel
    for channel in range(3):
        diff_map = lab_frame2[channel, :,:] - lab_frame1[channel,:,:]
        diff_map = diff_map**2
        diff_map = np.sqrt(diff_map)
 
    # Compute statistics for diff maps and diff values
    mean_color_diff = np.mean(diff_map) # per frame
    max_color_diff = np.max(diff_map)
    
    if(max_color_diff > max):
        max = max_color_diff
        max_diff_loc = np.unravel_index(np.argmax(diff_map), diff_map.shape)  # Location of max difference
    

    return mean_color_diff, max, max_diff_loc, diff_map


# def compute_color_difference(lab_frame1, lab_frame2):
#     """
#     Compute the color difference between two frames in LAB color space using Euclidean distance.

#     Parameters
#     ----------
#     lab_frame1 : numpy.array
#         First frame in LAB color space.
#     lab_frame2 : numpy.array
#         Second frame in LAB color space.

#     Returns
#     -------
#     mean_color_diff : float
#         Mean color difference across the image.
#     max_color_diff : float
#         Maximum color difference observed.
#     max_diff_loc : tuple
#         Location (row, col) of the maximum color difference.
#     diff_map : numpy.array
#         2D array of pixel-wise color differences.
#     """

#     # Compute Euclidean distance at each pixel
#     diff_map = np.sqrt(np.sum((lab_frame2 - lab_frame1) ** 2, axis=-1))

#     # Compute statistics for diff maps and diff values
#     mean_color_diff = np.mean(diff_map)  # per frame
#     max_color_diff = np.max(diff_map)
#     max_diff_loc = np.unravel_index(np.argmax(diff_map), diff_map.shape)  # Ensure it's always initialized

#     return mean_color_diff, max_color_diff, max_diff_loc, diff_map
