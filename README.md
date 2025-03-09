# CAMpare NR IQA - TNM097 Project
No-reference IQA on live video in OpenCV

## Objective metrics used: 
- SCIELAB
- NIQE
- BRISQUE

## Subjective metric used: 
- PaQ2PiQ

## Dependencies: 
- pip install opencv-python 
- pip install scipy
- pip install numpy 
- pip install skimage
- pip install pyica
- pip install tabulate

- 


## Files: 
- **main.py:** SPD constants WIDTH, HEIGHT, SCREEN_DIAGONAL can be changed here, MAX_SIZE for sequence size to process, and camera port initialization <br> The main file of the program, which captures video, evaluates video sequences and logs results 
- **scielab.py:** Converts BGR frames in a batch to opponent HVS filtered space, LAB space and computes average color difference on each frame in the batch with euclidean distance. Contains intermediate conversions steps to and from XYZ and Gaussian-based filtering for HVS. 
- **importedMetrics.py:** This file fetches automatic metrics from the pyiqa library and converts frames to pytorch tensors for compatibility.  
- **logger.py:** File that saves captured frames, evaluation results and color difference maps from main.py to directory. Can also print evaluation results to OpenCV-frames. 
- **scaleResolution.py:** Scales captured frames to 400X400 and cuts the top 40 pixels
- **SPD.py:** Computes the spd from values given in main.py. Defaults viewing distance to 40cm. 
- **FPS.py:** Computes and prints the fps rate of video capture. Print can also be put onto captured frames.

- **getAvailableCameras.py:** Code snippet that prints what camera ports are available on the machine. Assumes number of available ports is 5 or less.
- **testCams:** Code snippet to test found camera ports and camera positionings. Quit run by hitting "q". Defaulted to ports 0 and 1. 


## How to run: <br>
Preferably run within virtual python environment .venv in VSCode<br>
Within the code directory, run from commandline with: 
- python main.py
- python getAvailableCams.py
- python testCams.py


