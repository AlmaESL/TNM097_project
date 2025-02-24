import numpy as np 
import math

def compute_spd(screen_width, screen_height, screen_diagonal, viewing_distance=15.748): 
    
    """
    Compute the Samples Per Degree of a screen given its dimensions
    and viewing distance

    Parameters
    ----------
    screen_width : int
        Width of the screen in pixels
    screen_height : int
        Height of the screen in pixels
    screen_diagonal : float
        Diagonal of the screen in inches
    viewing_distance : float, optional
        Viewing distance in inches. Defaults to 15.748 = 40 cm

    Returns
    -------
    spd : float
        SPD value 
    """

    ppi = np.sqrt(screen_width**2 + screen_height**2) / screen_diagonal
    spd = round(ppi * viewing_distance * (math.pi/180), 3)
    print("\nComputing spd...")
    print("Samples Per Degree: ", spd, "\n")
    
    return spd
    