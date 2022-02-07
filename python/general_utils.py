from scipy.spatial.distance import cdist
import numpy as np

#### HL2 RM File Loading Utilities

def load_r2c(file_name):
    x = np.loadtxt(file_name, delimiter=',')
    x = np.reshape(x, (4, 4))
    return x

def load_r2w(file_name):
    return np.loadtxt(file_name, dtype=str, delimiter=',')

def load_lut(lut_filename):
    with open(lut_filename, mode='rb') as depth_file:
        lut = np.frombuffer(depth_file.read(), dtype="f")
        lut = np.reshape(lut, (-1, 3))
    return lut

#### Efficient 2-Phase Grid Search 

def search_intrinsics(norm, intrinsics, scale_factor=20):
        
    # subsample original grid
    iss = intrinsics[::scale_factor, ::scale_factor, :]
    issf = np.reshape(iss, (-1,norm.shape[1]))   

    # search in subsampled grid
    mins = cdist(norm, issf)
    mins = np.argmin(mins, axis=1)
    pixels = np.unravel_index(mins, iss.shape[:-1])
    pixels = np.array(pixels).transpose()
    pixels = pixels * scale_factor
    
    # create windowed focus area
    row_start = np.max([np.min(pixels[:,0] - (scale_factor + 5)), 0])
    row_end = np.min([np.max(pixels[:,0] + (scale_factor + 5)), intrinsics.shape[0] - 1])
    col_start = np.max([np.min(pixels[:,1] - (scale_factor + 5)), 0])
    col_end = np.min([np.max(pixels[:,1] + (scale_factor + 5)), intrinsics.shape[1] - 1])
    
    # search in focus area
    i_segment = intrinsics[row_start:row_end, col_start:col_end, :]
    i_segmentf = np.reshape(i_segment, (-1,norm.shape[1]))
    mins = cdist(norm, i_segmentf)
    #print(np.min(mins, 1))
    mins = np.argmin(mins, axis=1)
    pixels = np.unravel_index(mins, i_segment.shape[:-1])
    pixels = np.array(pixels).transpose()
    pixels += [row_start, col_start] # place back in big grid
    
    return pixels

#### plotting with slight robustness to outlier pixel values (written for vein)

def plot_conts(valid_pixels, bright, lims=0):
    canvas = np.zeros((512,512))
    canvas[valid_pixels[:,0], valid_pixels[:,1]] = bright
    if lims == 0:
        plt.imshow(canvas)
    else:
        minval = np.percentile(bright, 0.05)
        maxval = np.percentile(bright, 0.975)
        plt.imshow(canvas, vmin=minval, vmax=maxval)
    return canvas    