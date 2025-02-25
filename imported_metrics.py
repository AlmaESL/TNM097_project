import pyiqa #BRISQUE, NIQUE, PaQ2PiQ and A LOT more!
import numpy as np
import torch # make frames into tensors - compatible with pyiqa


# def preprocess_frames(frames_deque):

#     """
#     Preprocess frames from a deque for input to a PyTorch model.

#     Parameters
#     ----------
#     frames_deque : deque
#         A deque containing frames, each as a numpy array.

#     Returns
#     -------
#     torch.Tensor
#         A tensor of shape (N, C, H, W) with normalized values in [0, 1].
#     """
#     # Convert to NumPy array (N, H, W, C) and normalize to [0, 1]
#     frames = np.stack(frames_deque, axis=0)  
#     frames = frames.astype(np.float32) / 255.0 
    
#     # Convert to (N, C, H, W) and then to PyTorch tensor, return 
#     frames = np.transpose(frames, (0, 3, 1, 2)) 
#     return torch.tensor(frames)



# def compute_niqe_avg(frames_deque): 
#     """
#     Compute the average NIQE of frames in a deque

#     Parameters
#     ----------
#     frames_deque : deque
#         deque of frames

#     Returns
#     -------
#     avg_niqe : matrix
#         average NIQE of frames in the deque
#     """
    
#     # Initialize the NIQE metric
#     niqe_metric = pyiqa.create_metric('niqe').to('cpu')
    
#     frames_tensor = preprocess_frames(frames_deque)
#     return torch.mean(niqe_metric(frames_tensor)).item()
    
  


# def compute_brisque_avg(frames_deque): 
#     """
#     Compute the average BRISQUE of frames in a deque

#     Parameters
#     ----------
#     frames_deque : deque
#         deque of frames

#     Returns
#     -------
#     avg_brisque : matrix
#         average BRISQUE of frames in the deque
#     """
    
#     # Initialize the BRISQUE metric
#     brisque_metric = pyiqa.create_metric('brisque').to("cpu")
    
#     frames_tensor = preprocess_frames(frames_deque)
#     return torch.mean(brisque_metric(frames_tensor)).item()


# def compute_paq2piq_avg(frames_deque): 
#     """
#     Compute the average PaQ2PiQ of frames in a deque

#     Parameters
#     ----------
#     frames_deque : deque
#         deque of frames

#     Returns
#     -------
#     avg_paq2piq : matrix
#         average PaQ2PiQ of frames in the deque
#     """
    
#     # Initialize the PaQ2PiQ metric
#     paq2piq_metric = pyiqa.create_metric('paq2piq').to('cpu')

#     frames_tensor = preprocess_frames(frames_deque)
#     return torch.mean(paq2piq_metric(frames_tensor)).item()


# print("\nRetrieving imports...\n")
# niqe_metric = pyiqa.create_metric('niqe').to('cpu')
# brisque_metric = pyiqa.create_metric('brisque').to('cpu')
# paq2piq_metric = pyiqa.create_metric('paq2piq').to('cpu')

# def preprocess_frame(frame):
#     """
#     Preprocess a single frame for input to a PyTorch model.
#     """
#     frame = frame.astype(np.float32) / 255.0
#     frame = np.transpose(frame, (2, 0, 1))
#     return torch.tensor(frame).unsqueeze(0) 

# def compute_niqe(frame):
#     """Compute NIQE for a single frame."""
#     niqe_metric = pyiqa.create_metric('niqe').to('cpu')
#     frame_tensor = preprocess_frame(frame)
#     return niqe_metric(frame_tensor).item()

# def compute_brisque(frame):
#     """Compute BRISQUE for a single frame."""
#     brisque_metric = pyiqa.create_metric('brisque').to("cpu")
#     frame_tensor = preprocess_frame(frame)
#     return brisque_metric(frame_tensor).item()

# def compute_paq2piq(frame):
#     """Compute PaQ2PiQ for a single frame."""
#     paq2piq_metric = pyiqa.create_metric('paq2piq').to('cpu')
#     frame_tensor = preprocess_frame(frame)
#     return paq2piq_metric(frame_tensor).item()


niqe_metric = pyiqa.create_metric('niqe').to('cpu')
paq2piq_metric = pyiqa.create_metric('paq2piq').to('cpu')

def preprocess_frame(frame):
    frame = frame.astype(np.float32) / 255.0
    frame = np.transpose(frame, (2, 0, 1))
    return torch.tensor(frame).unsqueeze(0)

def compute_metrics(frame):
    """Compute all quality metrics for a single frame"""
    brisque_metric = pyiqa.create_metric('brisque').to('cpu')
    frame_tensor = preprocess_frame(frame)
    return {
        'niqe': niqe_metric(frame_tensor).item(),
        'brisque': brisque_metric(frame_tensor).item(),
        'paq2piq': paq2piq_metric(frame_tensor).item()
    }







































# import pyiqa
# import numpy as np
# import torch
# import threading

# class QualityMetrics:
#     """Efficient computation of NIQE, BRISQUE, and PaQ2PiQ in a separate thread"""

#     def __init__(self):
#         # Initialize PyIQA models once
#         self.niqe_model = pyiqa.create_metric('niqe').to("cpu")
#         self.brisque_model = pyiqa.create_metric('brisque').to("cpu")
#         self.paq2piq_model = pyiqa.create_metric('paq2piq').to("cpu")

#         # Store last computed values (avoids recomputation)
#         self.niqe_value = 0
#         self.brisque_value = 0
#         self.paq2piq_value = 0

#         # Lock for thread safety
#         self.lock = threading.Lock()

#     def preprocess_frames(self, frames_deque):
#         """Convert frames to (N, C, H, W) format for PyIQA"""
#         frames = np.stack(frames_deque, axis=0)  # Convert to NumPy array (N, H, W, C)
#         frames = frames.astype(np.float32) / 255.0  # Normalize to [0,1]
#         frames = np.transpose(frames, (0, 3, 1, 2))  # Convert to (N, C, H, W)
#         return torch.tensor(frames)

#     def compute_metrics_async(self, frames_deque):
#         """Runs NIQE, BRISQUE, and PaQ2PiQ asynchronously"""
#         def compute():
#             frames_tensor = self.preprocess_frames(frames_deque)

#             with self.lock:
#                 self.niqe_value = torch.mean(self.niqe_model(frames_tensor)).item()
#                 self.brisque_value = torch.mean(self.brisque_model(frames_tensor)).item()
#                 self.paq2piq_value = torch.mean(self.paq2piq_model(frames_tensor)).item()

#         # Run in a background thread
#         thread = threading.Thread(target=compute, daemon=True)
#         thread.start()

#     def get_metrics(self):
#         """Safely return the latest computed metrics"""
#         with self.lock:
# #             return self.niqe_value, self.brisque_value, self.paq2piq_value



# def preprocess_frames(frames_deque):
#     frames = np.stack(frames_deque, axis=0).astype(np.float32) / 255.0
#     frames = np.transpose(frames, (0, 3, 1, 2))
#     return torch.tensor(frames)

# def compute_metric_avg(frames_deque, metric_name, rolling_buffer):
#     metric = pyiqa.create_metric(metric_name).to('cpu')
#     frames_tensor = preprocess_frames(frames_deque)
#     new_value = torch.mean(metric(frames_tensor)).item()
    
#     if len(rolling_buffer) == MAX_FRAMES:
#         rolling_buffer.pop(0)
#     rolling_buffer.append(new_value)
    
#     return np.mean(rolling_buffer)