import pyiqa #BRISQUE, NIQUE, PaQ2PiQ and A LOT more!
import numpy as np
import torch # make frames into tensors - compatible with pyiqa


niqe_metric = pyiqa.create_metric('niqe').to('cpu')

paq2piq_metric = pyiqa.create_metric('paq2piq').to('cpu')

nima_metric = pyiqa.create_metric('nima').to('cpu')
piqe_metric = pyiqa.create_metric('piqe').to('cpu')

def preprocess_frame(frame):
    """
    Preprocess a frame by converting it to a PyTorch tensor and
    normalizing it to the range [0, 1].

    Parameters
    ----------
    frame : np.ndarray
        Input frame

    Returns
    -------
    preprocessed_frame : torch.Tensor
        Preprocessed frame as a PyTorch tensor
    """
    # Normalize frame
    frame = frame.astype(np.float32) / 255.0
    
    # Transpose frame
    frame = np.transpose(frame, (2, 0, 1))
    
    # Convert frame to PyTorch tensor
    return torch.tensor(frame).unsqueeze(0)

def compute_metrics(frame):

    """
    Compute a set of metrics for a given frame.

    Parameters
    ----------
    frame : np.ndarray
        Input frame

    Returns
    -------
    metrics : dict
        Dictionary of metrics, with keys 'niqe', 'brisque', 'paq2piq', 'nima', and 'piqe'
    """
    brisque_metric = pyiqa.create_metric('brisque').to('cpu')
    frame_tensor = preprocess_frame(frame)
    return {
        
        'niqe': niqe_metric(frame_tensor).item(),
        
        'brisque': brisque_metric(frame_tensor).item(),
        'paq2piq': paq2piq_metric(frame_tensor).item(),
        'nima': nima_metric(frame_tensor).item(), 
        'piqe': piqe_metric(frame_tensor).item()
    }

