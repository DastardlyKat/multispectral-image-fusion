import cv2
import numpy as np
import os

## Loading the image folders from the device
vis_path = '/Users/swaksharbora/Documents/ML Projects/multispectral_fusion/LLVIP/visible/train'
ir_path = '/Users/swaksharbora/Documents/ML Projects/multispectral_fusion/LLVIP/infrared/train'

## Getting the sorted image files from the folders
file_a = sorted(os.listdir(vis_path))
file_b = sorted(os.listdir(ir_path))

def preprocess(vis_path, ir_path, limit=20):
    file_list = sorted(os.listdir(vis_path))[:limit]
    preprocessed_pairs = []
    
    ## Reading the image files from the folder
    for file in file_list:
        vis_image = cv2.imread(os.path.join(vis_path, file))
        ir_image = cv2.imread(os.path.join(ir_path, file))
    
        # Skip if either file is not present
        if vis_image is None or ir_image is None:
            print(f"Skipping {file} - Could not load")
            continue
    
        ## Converting the images into grayscale images
        vis_gray = cv2.cvtColor(vis_image, cv2.COLOR_BGR2GRAY)
        ir_gray = cv2.cvtColor(ir_image, cv2.COLOR_BGR2GRAY)
    
        ## Checking dimensions
        if vis_gray.shape != ir_gray.shape:
            vis_gray = cv2.resize(vis_gray, (ir_gray.shape[1], ir_gray.shape[0]))         ## If different size, then make their sizes same
            
        # Histogram Equalisation
        vis_eq = cv2.equalizeHist(vis_gray)
        ir_eq = cv2.equalizeHist(ir_gray)
    
        ## Normalisation
        vis_normalised = vis_eq.astype(np.float64) / 255.0
        ir_normalised = ir_eq.astype(np.float64) / 255.0
        
        preprocessed_pairs.append((file, vis_normalised, ir_normalised))
    
    ## Returning the normalised images
    return preprocessed_pairs

if __name__ ==  "__main__":
    pairs = preprocess(vis_path, ir_path)
    print(f"Total pairs loaded: {len(pairs)}")
    print(f"First pair filename: {pairs[0][0]}")
    print(f"Visible image shape: {pairs[0][1].shape}")
    print(f"Infrared image shape: {pairs[0][2].shape}")
    print(f"Visible value range: {pairs[0][1].min():.3f} - {pairs[0][1].max():.3f}")
    print(f"Infrared value range: {pairs[0][2].min():.3f} - {pairs[0][2].max():.3f}")