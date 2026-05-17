from preprocessing import preprocess, vis_path, ir_path

import numpy as np
import pywt

## Simple Average Fusion
def simpleAverageFusion(preprocessed_pairs):
    average_fusion = []
    
    for file, vis, ir in preprocessed_pairs:
        ## Average Fusion
        fused = (vis + ir) / 2.0
        
        fused = np.clip(fused, 0, 1)
        
        average_fusion.append((file, fused))
        
    return average_fusion

## PCA Fusion
def principleComponentAnalysis(preprocessed_pairs):
    pca_fusion = []
    
    for file, vis, ir in preprocessed_pairs:
        
        ## Flatten images
        vis_flat = vis.flatten()
        ir_flat = ir.flatten()
        
        # Stack images into a 2-D array
        array = np.stack((vis_flat, ir_flat), axis=0)
        
        cov = np.cov(array)
        
        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        # Get weights
        weights = eigenvectors[:, np.argmax(eigenvalues)]
        weights = np.abs(weights) / np.sum(np.abs(weights))     # Normalising the sum to 1

        # Weigthed sum
        fused = weights[0] * vis + weights[1] * ir
        
        # Clipping to [0,1]
        fused = np.clip(fused, 0, 1)
        
        pca_fusion.append((file, fused))
        
    return pca_fusion

## DWT Fusion
def discreteWaveletTransform(preprocessed_pairs):
    dwt_fusion = []
    
    for file, vis, ir in preprocessed_pairs:
        # Perfrom 2D DWT
        LL1, (LH1, HL1, HH1) = pywt.dwt2(vis, 'haar')
        LL2, (LH2, HL2, HH2) = pywt.dwt2(ir, 'haar')
        
        # Fusion rules
        LL = 0.5 * LL1 + 0.5 * LL2
        LH = np.where(np.abs(LH1) >= np.abs(LH2), LH1, LH2)
        HL = np.where(np.abs(HL1) >= np.abs(HL2), HL1, HL2)
        HH = np.where(np.abs(HH1) >= np.abs(HH2), HH1, HH2)
        
        # Inverse DWT to get fused image
        fused = pywt.idwt2((LL, (LH, HL, HH)), 'haar')
        
        fused = np.clip(fused, 0, 1)
        
        dwt_fusion.append((file, fused))
        
    return dwt_fusion

if __name__ == "__main__":
    pairs = preprocess(vis_path, ir_path, limit=5)
    
    avg_results = simpleAverageFusion(pairs)
    pca_results = principleComponentAnalysis(pairs)
    dwt_results = discreteWaveletTransform(pairs)
    
    print(f"Average fusion results: {len(avg_results)}")
    print(f"PCA fusion results: {len(pca_results)}")
    print(f"DWT fusion results: {len(dwt_results)}")
    
    print(f"Sample fused image shape: {dwt_results[0][1].shape}")
    print(f"Sample value range: {dwt_results[0][1].min():.3f} - {dwt_results[0][1].max():.3f}")
