import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from preprocessing import preprocess, vis_path, ir_path
from fusion import simpleAverageFusion, principleComponentAnalysis, discreteWaveletTransform

## Entropy
def compute_entropy(fused):
    fused_uint8 = (fused * 255).astype(np.uint8)
    histogram, _ = np.histogram(fused_uint8, bins=256, range=(0, 255))
    histogram = histogram / histogram.sum()
    entropy = -np.sum([p * np.log2(p) for p in histogram if p > 0])
    return entropy

## Spatial Frequency
def compute_spatial_frequency(fused):
    M, N = fused.shape                                  # Get dimensions
    row_diff = fused[:, 1:] - fused[:, :-1]             # Row-wise differences
    RF = np.sqrt(np.sum(row_diff ** 2) / (M * N))       # Row frequency
    col_diff = fused[1:, :] - fused[:-1, :]             # Column-wise differences
    CF = np.sqrt(np.sum(col_diff ** 2) / (M * N))       # Column frequency
    SF = np.sqrt(RF ** 2 + CF ** 2)                     # Overall spatial frequency
    return SF

## SSIM
def compute_ssim(fused, vis, ir):
    ssim_vis = ssim(fused, vis, data_range=1.0)         # SSIM with visible image
    ssim_ir = ssim(fused, ir, data_range=1.0)           # SSIM with infrared image
    return ssim_vis, ssim_ir

## PSNR
def compute_psnr(fused, vis, ir):
    psnr_vis = psnr(vis, fused, data_range=1.0)         # PSNR with visible image
    psnr_ir = psnr(ir, fused, data_range=1.0)           # PSNR with infrared image
    return psnr_vis, psnr_ir

## Edge Intensity
def compute_edge_intensity(fused):
    fused_uint8 = (fused * 255).astype(np.uint8)                        # Converting to uint8 for edge detection
    sobel_x = cv2.Sobel(fused_uint8, cv2.CV_64F, 1, 0, ksize=3)         # Sobel operator in x direction
    sobel_y = cv2.Sobel(fused_uint8, cv2.CV_64F, 0, 1, ksize=3)         # Sobel operator in y direction
    gradient_magnitude = cv2.magnitude(sobel_x, sobel_y)                # Gradient magnitude
    EI = np.mean(gradient_magnitude)                                    # Average edge intensity across the image
    return EI

## Compute all metrics for a set of fusion results
def compute_all_metrics(fusion_results, preprocessed_pairs):
    source_lookup = {file: (vis, ir) for file, vis, ir in preprocessed_pairs}
    all_metrics = []
    for file, fused in fusion_results:
        vis, ir = source_lookup[file]
        EN = compute_entropy(fused)
        SF = compute_spatial_frequency(fused)
        ssim_vis, ssim_ir = compute_ssim(fused, vis, ir)
        psnr_vis, psnr_ir = compute_psnr(fused, vis, ir)
        EI = compute_edge_intensity(fused)
        all_metrics.append({
            'filename': file,
            'entropy': EN,
            'spatial_frequency': SF,
            'ssim_vis': ssim_vis,
            'ssim_ir': ssim_ir,
            'psnr_vis': psnr_vis,
            'psnr_ir': psnr_ir,
            'edge_intensity': EI
        })
    return all_metrics

## Print averaged summary across all images for a method
def summarize_metrics(all_metrics, method_name):
    keys = ['entropy', 'spatial_frequency', 'ssim_vis', 'ssim_ir', 'psnr_vis', 'psnr_ir', 'edge_intensity']
    print(f"\n--- {method_name} ---")
    for key in keys:
        avg = np.mean([m[key] for m in all_metrics])
        print(f"  {key}: {avg:.4f}")

if __name__ == "__main__":
    pairs = preprocess(vis_path, ir_path, limit=5)
    avg_results = simpleAverageFusion(pairs)
    pca_results = principleComponentAnalysis(pairs)
    dwt_results = discreteWaveletTransform(pairs)
    avg_metrics = compute_all_metrics(avg_results, pairs)
    pca_metrics = compute_all_metrics(pca_results, pairs)
    dwt_metrics = compute_all_metrics(dwt_results, pairs)
    summarize_metrics(avg_metrics, "Simple Average Fusion")
    summarize_metrics(pca_metrics, "PCA Fusion")
    summarize_metrics(dwt_metrics, "Wavelet DWT Fusion")