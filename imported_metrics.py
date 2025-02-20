import pyiqa #BRISQUE, NIQUE, PaQ2PiQ and A LOT more!
import numpy as np

def compute_niqe_avg(frames_deque): 
    """
    Compute the average NIQE of frames in a deque

    Parameters
    ----------
    frames_deque : deque
        deque of frames

    Returns
    -------
    avg_niqe : matrix
        average NIQE of frames in the deque
    """

    frames_deque = np.stack(frames_deque, axis=0)  
    return np.mean(pyiqa.compute_metric(frames_deque, 'niqe'))


def compute_brisque_avg(frames_deque): 
    """
    Compute the average BRISQUE of frames in a deque

    Parameters
    ----------
    frames_deque : deque
        deque of frames

    Returns
    -------
    avg_brisque : matrix
        average BRISQUE of frames in the deque
    """

    frames_deque = np.stack(frames_deque, axis=0)  
    return np.mean(pyiqa.compute_metric(frames_deque, 'brisque'))


def compute_paq2piq_avg(frames_deque): 
    """
    Compute the average PaQ2PiQ of frames in a deque

    Parameters
    ----------
    frames_deque : deque
        deque of frames

    Returns
    -------
    avg_paq2piq : matrix
        average PaQ2PiQ of frames in the deque
    """

    frames_deque = np.stack(frames_deque, axis=0)  
    return np.mean(pyiqa.compute_metric(frames_deque, 'paq2piq'))