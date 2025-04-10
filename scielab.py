import numpy as np
import cv2
from scipy.ndimage import convolve
from skimage.color import rgb2xyz


def rgb_to_xyz(image):
    """Convert RGB to XYZ"""
    return rgb2xyz(image) 


def xyz_to_lab(image, whitepoint=None, exp=1/3):
    
    """
    Convert an image from CIE XYZ color space to CIE LAB color space.

    Parameters:
    image -- input image in XYZ color space 
    whitepoint -- reference white point (defaults to D65: [95.05, 100, 108.88])
    exp -- exponent used for gamma correction (default: 1/3)

    Returns:
    lab -- image in LAB color space 
    """

    # D65
    if whitepoint is None:
        whitepoint = [95.05, 100, 108.88]

    Xn, Yn, Zn = whitepoint

    x = image[..., 0] / Xn
    y = image[..., 1] / Yn
    z = image[..., 2] / Zn

    # Find out points < 0.008856
    xx = x <= 0.008856
    yy = y <= 0.008856
    zz = z <= 0.008856
    
    lab = np.zeros_like(image)
    
    # Compute L* values
    fy = y[yy]
    y = np.abs(y)**exp
    lab[..., 0] = 116 * y - 16
    lab[yy, 0] = 903.3 * fy

    # Compute a* and b* values
    fx = 7.787 * x[xx] + 16 / 116
    fy = 7.787 * fy + 16 / 116
    fz = 7.787 * z[zz] + 16 / 116
    x = x**exp
    z = np.abs(z)**exp
    x[xx] = fx
    y[yy] = fy
    z[zz] = fz

    lab[..., 1] = 500 * (x - y)
    lab[..., 2] = 200 * (y - z)

    return lab
    
   

def xyz_to_opponent(XYZ):
    """
    Convert XYZ color space to opponent color space.

    Parameters
    ----------
    XYZ : array
        XYZ color space image.

    Returns
    -------
    O1, O2, O3 : array, each shape (M,N)
        Opponent color space images.
    """
    
    X, Y, Z = XYZ[:,:,0], XYZ[:,:,1], XYZ[:,:,2]
    
    # Matrix to convert XYZ to opponent space
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




def sum_gauss(params, width, dimension=1):
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
    dimension : int, optional
        Specifies whether the required sum of Gaussians is 1-D or 2-D. Defaults to 1.

    Returns
    -------
    g : numpy.array
        An array representing the summed Gaussian filter of specified width.
    """

    num_G = (len(params) - 1) // 2

    if dimension == 2:
        g = np.zeros((width, width))
    else:
        g = np.zeros(width)
    
    for i in range(num_G):
        half_width = params[2*i+1]
        weight = params[2*i+2]
        if dimension == 2:
            g0 = gauss2(half_width, width)
        else:
            g0 = gauss(half_width, width)
        g += weight * g0
    
    return g

def gauss2(half_width, width):
    """
    Return a 2-D Gaussian filter of specified half width and width.

    Parameters
    ----------
    half_width : int
        Half width of the Gaussian filter.
    width : int
        Width of the output filter.

    Returns
    -------
    gauss_filter : 2D array
        2-D Gaussian filter of specified width and half width.
    """
    alpha = 2 * np.sqrt(np.log(2)) / (half_width - 1)
    x = np.arange(width) - width // 2
    y = np.arange(width) - width // 2
    X, Y = np.meshgrid(x, y)
    gauss_filter = np.exp(-alpha**2 * (X**2 + Y**2))
    return gauss_filter / np.sum(gauss_filter)


def conv2(x, y, mode='same'):
    """
    2D convolution of two arrays.
    
    Parameters
    ----------
    x : numpy.array
        Input array to be convolved.
    y : numpy.array
        Convolution kernel.
    mode : str, optional
        Convolution mode, by default 'same'.
    
    Returns
    -------
    numpy.array
        Convolved array.
    """
    return convolve(x, y, mode=mode)



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
    
    # Parameters for the Gaussian kernels - from lab3 code
    x1 = [width, 0.05, 1.00327, 0.225, 0.114416, 7.0, -0.117686]
    x2 = [width, 0.0685, 0.616725, 0.826, 0.383275]
    x3 = [width, 0.0920, 0.567885, 0.6451, 0.432115]
    
    x1[1::2] = [x * spd for x in x1[1::2]]
    x2[1::2] = [x * spd for x in x2[1::2]]
    x3[1::2] = [x * spd for x in x3[1::2]]

    k1 = sum_gauss(x1, width, dimension=2).astype(np.float64)
    k2 = sum_gauss(x2, width, dimension=2).astype(np.float64)
    k3 = sum_gauss(x3, width, dimension=2).astype(np.float64)

    # Convolve with reflection padding 
    O1_f = conv2(O1, k1, mode='reflect')
    O2_f = conv2(O2, k2, mode='reflect')
    O3_f = conv2(O3, k3, mode='reflect')

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
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    xyz_frame = rgb_to_xyz(frame_rgb)
    
    # Step 2: Convert XYZ to opponent space
    O1, O2, O3 = xyz_to_opponent(xyz_frame)
    # print("O1: ", O1)
    # print("O2: ", O2)
    # print("O3: ", O3)
    
    # Step 3: Apply spatial filtering to the opponent space -> this represents the human visual system
    O1_f, O2_f, O3_f = apply_spatial_filter(O1, O2, O3, spd)
    # print("O1_f: ", O1_f)
    # print("O2_f: ", O2_f)
    # print("O3_f: ", O3_f)
    
    # Make matrix of o1_f, o2_f and o3_f
    opponent_space = np.array([O1_f, O2_f, O3_f])
  
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
 
    # Unpack the opponent channels
    O1 = frame[0, ...]
    O2 = frame[1, ...]
    O3 = frame[2, ...]
    
    # Apply the inverse transformation back to XYZ
    X = 0.610 * O1 - 1.819 * O2 - 0.149 * O3
    Y = 1.416 * O1 + 0.798 * O2 + 0.425 * O3
    Z = 1.772 * O1 + 0.628 * O2 + 2.471 * O3

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
    
    xyz_matrix = opponent_to_xyz(frame)
    lab_matrix = xyz_to_lab(xyz_matrix)
    
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
    
    # Transpose for logging color diff maps
    if lab_frame1.shape[0] != 3:
        lab_frame1 = np.transpose(lab_frame1, (2, 1, 0))
     
    if lab_frame2.shape[0] != 3:
        lab_frame2 = np.transpose(lab_frame2, (2, 1, 0))
    
    
    max = 0   
    # Compute color difference with Euclidean distance 
    diff_map = np.sqrt((lab_frame2[0,:,:] - lab_frame1[0,:,:])**2 + (lab_frame2[1,:,:] - lab_frame1[1,:,:])**2 + (lab_frame2[2,:,:] - lab_frame1[2,:,:])**2)
    
    # Compute statistics for diff maps and diff values
    mean_color_diff = np.mean(diff_map)
    max_color_diff = np.max(diff_map)
    
    # Locate location of max difference
    if(max_color_diff > max):
        max = max_color_diff
        max_diff_loc = np.unravel_index(np.argmax(diff_map), diff_map.shape)  
    

    return mean_color_diff, max, max_diff_loc, diff_map

